import os
import logging
import sys
import numpy as np
import rospy
import torch
import socket
import torch.nn as nn
from mpi4py import MPI
from torch.optim import Adam
from torch.autograd import Variable
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import time
import pandas as pd

from model.net3 import MLPPolicy, CNNPolicy
from model.net2 import EnhancedCNNPolicy
from stage_world2 import StageWorld
from model.ppo import ppo_update_stage2, generate_train_data
from model.ppo import generate_action, transform_buffer
from model.utils import get_group_terminal, get_filter_index

# Hyperparameters (新增学习率衰减参数)
MAX_EPISODES = 6000
LASER_BEAM = 512
LASER_HIST = 3
HORIZON = 128
GAMMA = 0.99
LAMDA = 0.95
BATCH_SIZE = 512
EPOCH = 4
COEFF_ENTROPY = 5e-4
CLIP_VALUE = 0.1
NUM_ENV = 24
OBS_SIZE = 512
ACT_SIZE = 2
INIT_LR = 5e-5      # 初始学习率
MIN_LR = 1e-5       # 最小学习率
LR_DECAY_STEPS = 2000  # 衰减步长

def run(comm, env, policy, policy_path, action_bound, optimizer,scheduler=None):
    rate = rospy.Rate(40)
    buff = []
    global_update = 0
    global_step = 0

    # 新增训练指标记录字典
    train_stats = {
        'value_loss': [],
        'policy_loss': [],
        'entropy': [],
        'learning_rate': [],
        'episode_rewards': [],
        'global_steps': []
    }

    start_time = time.time()

    if env.index == 0:
        writer = SummaryWriter(log_dir=os.path.join(policy_path, 'tensorboard'))
        env.reset_world()

    # 新增课程学习参数
    success_history = deque(maxlen=100)  # 记录最近100个episode的成功率
    difficulty_update_freq = 10  # 每10个episode更新难度
    
    # 新增训练统计
    episode_times = []
    success_rates = []

    for id in range(MAX_EPISODES):
        env.reset_pose()
        env.generate_goal_point()
        group_terminal = False
        ep_reward = 0
        liveflag = True
        step = 1
        episode_start = time.time()

        # 初始化动作缓冲区
        if hasattr(env, 'last_action'):
            env.last_action = np.zeros(ACT_SIZE)  # 根据动作维度初始化

        obs = env.get_laser_observation()
        obs_stack = deque([obs] * LASER_HIST)  # 初始化为6帧相同数据
        goal = np.asarray(env.get_local_goal())
        speed = np.asarray(env.get_self_speed())
        state = [obs_stack, goal, speed]

        while not group_terminal and not rospy.is_shutdown():
            state_list = comm.gather(state, root=0)

            # generate actions at rank==0
            v, a, logprob, scaled_action = generate_action(env=env, state_list=state_list,
                                                           policy=policy, action_bound=action_bound)
            # execute actions
            real_action = comm.scatter(scaled_action, root=0)
            if liveflag == True:
                env.control_vel(real_action)
                # rate.sleep()
                rospy.sleep(0.001)
                # 修改奖励获取接口，传递动作信息
                r, terminal, result = env.get_reward_and_terminate(step, real_action)  # 添加动作参数
                step += 1

            if liveflag == True:
                ep_reward += r
            if terminal == True:
                liveflag = False

            global_step += 1

            # 每步更新队列
            s_next = env.get_laser_observation()
            obs_stack.popleft()  # 移除最早的一帧
            obs_stack.append(s_next)  # 添加最新一帧
            goal_next = np.asarray(env.get_local_goal())
            speed_next = np.asarray(env.get_self_speed())
            state_next = [obs_stack, goal_next, speed_next]

            if global_step % HORIZON == 0:
                state_next_list = comm.gather(state_next, root=0)
                last_v, _, _, _ = generate_action(env=env, state_list=state_next_list, policy=policy,
                                                  action_bound=action_bound)
            # add transitons in buff and update policy
            r_list = comm.gather(r, root=0)
            terminal_list = comm.gather(terminal, root=0)

            terminal_list = comm.bcast(terminal_list, root=0)
            group_terminal = get_group_terminal(terminal_list, env.index)
            # ================= 课程学习更新 =================
            episode_duration = time.time() - episode_start
            episode_times.append(episode_duration)
            
            # 收集所有环境的结果
            all_results = comm.gather(result, root=0)
            if env.index == 0:
                # 更新学习率（修改为更稳定的衰减方式）
                if scheduler is not None:
                    scheduler.step()
                    current_lr = optimizer.param_groups[0]['lr']
                    writer.add_scalar('Learning Rate', current_lr, id)
                    # logger.info(f"Episode {id}: Learning rate adjusted to {current_lr:.2e}")

                # 计算成功率
                success_count = sum(1 for res in all_results if res == 'Reach Goal')
                current_success_rate = success_count / len(all_results)
                success_history.append(current_success_rate)

                # 每10个episode更新难度
                if id % difficulty_update_freq == 0 and id != 0:
                    avg_success = np.mean(success_history)
                    # 更新所有环境的难度参数
                    env.update_difficulty(avg_success)
                    # 广播新参数
                    new_params = (env.current_safe_distance, env.current_goal_size)
                else:
                    new_params = None
                buff.append((state_list, a, r_list, terminal_list, logprob, v))
                if len(buff) > HORIZON - 1:
                    s_batch, goal_batch, speed_batch, a_batch, r_batch, d_batch, l_batch, v_batch = \
                        transform_buffer(buff=buff)
                    filter_index = get_filter_index(d_batch)
                    # print len(filter_index)
                    t_batch, advs_batch = generate_train_data(rewards=r_batch, gamma=GAMMA, values=v_batch,
                                                              last_value=last_v, dones=d_batch, lam=LAMDA)
                    memory = (s_batch, goal_batch, speed_batch, a_batch, l_batch, t_batch, v_batch, r_batch, advs_batch)
                    # 执行PPO更新并获取返回的指标
                    value_loss, policy_loss, entropy = ppo_update_stage2(
                        policy=policy, optimizer=optimizer, batch_size=BATCH_SIZE, memory=memory,
                        filter_index=filter_index, epoch=EPOCH, coeff_entropy=COEFF_ENTROPY,
                        clip_value=CLIP_VALUE, num_step=HORIZON, num_env=NUM_ENV,
                        frames=LASER_HIST, obs_size=OBS_SIZE, act_size=ACT_SIZE,
                        return_metrics=True
                    )

                    train_stats['value_loss'].append(value_loss)
                    train_stats['policy_loss'].append(policy_loss)
                    train_stats['entropy'].append(entropy)
                    train_stats['learning_rate'].append(optimizer.param_groups[0]['lr'])

                    logger.info(
                        f"Update {global_update}: Value Loss={value_loss:.4f}, Policy Loss={policy_loss:.4f}, Entropy={entropy:.4f}")

                    if global_update % 10 == 0:
                        writer.add_scalar('Loss/Value', value_loss, global_step)
                        writer.add_scalar('Loss/Policy', policy_loss, global_step)
                        writer.add_scalar('Entropy', entropy, global_step)

                    buff = []
                    global_update += 1
            else:
                new_params = None
            # 广播新参数到所有进程
            new_params = comm.bcast(new_params, root=0)
            if new_params is not None:
                env.current_safe_distance, env.current_goal_size = new_params

            state = state_next

        if env.index == 0:
            # 记录成功率
            success_rates.append(current_success_rate)
            writer.add_scalar('Training/Success_Rate', current_success_rate, id)
            
            # 记录episode时间
            avg_episode_time = np.mean(episode_times[-10:])
            writer.add_scalar('Time/Episode', avg_episode_time, id)
            
            # 记录课程难度
            writer.add_scalar('Curriculum/Safe_Distance', env.current_safe_distance, id)
            writer.add_scalar('Curriculum/Goal_Size', env.current_goal_size, id)

            # ================= 修改模型保存策略 =================
            if global_update != 0 and global_update % 20 == 0:
                torch.save(policy.state_dict(), policy_path + '/stage2_{}.pth'.format(global_update))
                logger.info('########################## model saved when update {} times##########################'.format(
                    global_update))
                train_stats['global_steps'].append(global_step)
                train_stats['episode_rewards'].append(ep_reward)
                logger.info(f"Training Stats: Step={global_step}, Reward={ep_reward:.2f}")
                writer.add_scalar('Reward/Episode', ep_reward, global_step)
                elapsed_time = time.time() - start_time
                logger.info(
                    f"Elapsed Time: {elapsed_time:.2f}s, Steps per Second: {global_step / elapsed_time:.2f}")

        logger.info('Env %02d, Goal (%05.1f, %05.1f), Episode %05d, setp %03d, Reward %-5.1f, %s,' % \
                    (env.index, env.goal_point[0], env.goal_point[1], id, step - 1, ep_reward, result))
        logger_cal.info(ep_reward)

    # 训练结束后保存统计数据
    if env.index == 0:
        # 保存最终课程学习曲线
        pd.DataFrame({
            'episode': range(len(success_rates)),
            'success_rate': success_rates,
            'safe_distance': [env.base_safe_distance*(0.9**(i//10)) for i in range(len(success_rates))],
            'goal_size': [env.base_goal_size*(0.95**(i//10)) for i in range(len(success_rates))]
        }).to_csv(os.path.join(policy_path, 'curriculum_progress.csv'), index=False)
        # df = pd.DataFrame(train_stats)
        # df.to_csv(os.path.join(policy_path, 'training_stats.csv'), index=False)
        writer.close()




if __name__ == '__main__':

    # config log
    hostname = socket.gethostname()
    if not os.path.exists('./log/' + hostname):
        os.makedirs('./log/' + hostname)
    output_file = './log/' + hostname + '/output.log'
    cal_file = './log/' + hostname + '/cal.log'

    # config log
    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(output_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    logger_cal = logging.getLogger('loggercal')
    logger_cal.setLevel(logging.INFO)
    cal_f_handler = logging.FileHandler(cal_file, mode='a')
    file_handler.setLevel(logging.INFO)
    logger_cal.addHandler(cal_f_handler)


    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    # 验证MPI进程数匹配
    assert size == NUM_ENV, f"MPI进程数({size})必须等于NUM_ENV({NUM_ENV})"

    

    env = StageWorld(512, index=rank, num_env=NUM_ENV)
    reward = None
    action_bound = [[-0.5, -1], [2, 1]]
    # torch.manual_seed(1)
    # np.random.seed(1)

    # 初始化课程学习参数
    if rank == 0:
        initial_params = (env.base_safe_distance, env.base_goal_size)
    else:
        initial_params = None
        
    # 广播初始参数
    initial_params = comm.bcast(initial_params, root=0)
    env.current_safe_distance, env.current_goal_size = initial_params

    if rank == 0:
        policy_path = 'policy/policy_ori'
        # policy = MLPPolicy(obs_size, act_size)

        # # 创建原始网络和改进后的网络
        # original_cnn_policy = OriginalCNNPolicy(frames=3, action_space=2)
        # # 加载权重
        # original_cnn_policy.load_state_dict(torch.load('policy/policy_ori/stage2_840.pth'))
        
        # new_cnn_policy = CNNPolicy(frames=3, action_space=2)
        # new_cnn_policy = initialize_with_original_weights(new_cnn_policy, original_cnn_policy)

        policy = CNNPolicy(frames=LASER_HIST, action_space=ACT_SIZE)
        policy.cuda()
        # 初始化优化器并添加学习率衰减
        opt = Adam(policy.parameters(), lr=INIT_LR)
        
        # 使用阶梯式指数衰减（更稳定的衰减方式）
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=opt,
            step_size=LR_DECAY_STEPS,
            gamma=0.5,
            last_epoch=-1
        )
        # 添加学习率下限保护
        scheduler = torch.optim.lr_scheduler.ChainedScheduler([
            scheduler,
            torch.optim.lr_scheduler.ConstantLR(
                opt, 
                factor=1.0,
                total_iters=MAX_EPISODES//LR_DECAY_STEPS*LR_DECAY_STEPS,
            )
        ])
        mse = nn.MSELoss()
        # 添加余弦退火学习率调度（MAX_EPISODES为总训练轮次）
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=MAX_EPISODES, eta_min=1e-5
        )

        if not os.path.exists(policy_path):
            os.makedirs(policy_path)

        file = policy_path + '/stage2.pth'
        if os.path.exists(file):
            logger.info('####################################')
            logger.info('############Loading Model###########')
            logger.info('####################################')
            state_dict = torch.load(file)
            policy.load_state_dict(state_dict)
        else:
            logger.info('#####################################')
            logger.info('############Start Training###########')
            logger.info('#####################################')
    else:
        policy = None
        policy_path = None
        opt = None


    try:
        if rank == 0:
            run(comm=comm, env=env, policy=policy, policy_path=policy_path, 
            action_bound=action_bound, optimizer=opt, scheduler=scheduler)
        else:
            run(comm=comm, env=env, policy=policy, policy_path=policy_path,
                action_bound=action_bound, optimizer=opt, scheduler=None)
    except KeyboardInterrupt:
        import traceback
        traceback.print_exc()
