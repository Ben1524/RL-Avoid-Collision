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
from torch.utils.tensorboard import SummaryWriter  # 新增

from model.net3 import CNNPolicy
from stage_world3 import StageWorld
from model.ppo import  generate_train_data
from model.ppo import generate_action, transform_buffer
from model.utils import get_group_terminal, get_filter_index
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

MAX_EPISODES = 5000
LASER_BEAM = 512
LASER_HIST = 6
HORIZON = 128
GAMMA = 0.99
LAMDA = 0.95
BATCH_SIZE = 512
EPOCH = 10
COEFF_ENTROPY = 5e-4
CLIP_VALUE = 0.10
NUM_ENV = 24
OBS_SIZE = 512
ACT_SIZE = 2
LEARNING_RATE = 5e-5



def ppo_update_stage2(policy, optimizer, batch_size, memory, filter_index, epoch,
                      coeff_entropy=0.02, clip_value=0.2,
                      num_step=2048, num_env=12, frames=1, obs_size=24, act_size=4,
                      return_metrics=False, writer=None, global_train_step=0):  # 新增writer和global_train_step
    obss, goals, speeds, actions, logprobs, targets, values, rewards, advs = memory

    advs = (advs - advs.mean()) / advs.std()

    obss = obss.reshape((num_step * num_env, frames, obs_size))
    goals = goals.reshape((num_step * num_env, 2))
    speeds = speeds.reshape((num_step * num_env, 2))
    actions = actions.reshape(num_step * num_env, act_size)
    logprobs = logprobs.reshape(num_step * num_env, 1)
    advs = advs.reshape(num_step * num_env, 1)
    targets = targets.reshape(num_step * num_env, 1)

    obss = np.delete(obss, filter_index, 0)
    goals = np.delete(goals, filter_index, 0)
    speeds = np.delete(speeds, filter_index, 0)
    actions = np.delete(actions, filter_index, 0)
    logprobs = np.delete(logprobs, filter_index, 0)
    advs = np.delete(advs, filter_index, 0)
    targets = np.delete(targets, filter_index, 0)

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    batch_count = 0

    for update in range(epoch):
        sampler = BatchSampler(SubsetRandomSampler(list(range(advs.shape[0]))),
                               batch_size=batch_size, drop_last=True)
        for i, index in enumerate(sampler):
            sampled_obs = Variable(torch.from_numpy(obss[index])).float().cuda()
            sampled_goals = Variable(torch.from_numpy(goals[index])).float().cuda()
            sampled_speeds = Variable(torch.from_numpy(speeds[index])).float().cuda()

            sampled_actions = Variable(torch.from_numpy(actions[index])).float().cuda()
            sampled_logprobs = Variable(torch.from_numpy(logprobs[index])).float().cuda()
            sampled_targets = Variable(torch.from_numpy(targets[index])).float().cuda()
            sampled_advs = Variable(torch.from_numpy(advs[index])).float().cuda()

            new_value, new_logprob, dist_entropy = policy.evaluate_actions(
                sampled_obs, sampled_goals, sampled_speeds, sampled_actions)

            sampled_logprobs = sampled_logprobs.view(-1, 1)
            ratio = torch.exp(new_logprob - sampled_logprobs)

            sampled_advs = sampled_advs.view(-1, 1)
            surrogate1 = ratio * sampled_advs
            surrogate2 = torch.clamp(ratio, 1 - clip_value, 1 + clip_value) * sampled_advs
            policy_loss = -torch.min(surrogate1, surrogate2).mean()

            sampled_targets = sampled_targets.view(-1, 1)
            value_loss = nn.functional.mse_loss(new_value, sampled_targets)

            loss = policy_loss + 20 * value_loss - coeff_entropy * dist_entropy
            optimizer.zero_grad()
            loss.backward()

            # 在优化步骤前添加梯度裁剪
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=5.0)
            optimizer.step()

            # 累积指标
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += dist_entropy.item()
            batch_count += 1

            # 记录每批次日志
            logging.getLogger('mylogger').info('Policy Loss: {:.4f}, Value Loss: {:.4f}, Entropy: {:.4f}'.format(
                policy_loss.item(), value_loss.item(), dist_entropy.item()))

            # 新增：记录当前批次指标到TensorBoard（仅Rank=0）
            if writer is not None:
                writer.add_scalar('Train/Policy_Loss', policy_loss.item(), global_train_step)
                writer.add_scalar('Train/Value_Loss', value_loss.item(), global_train_step)
                writer.add_scalar('Train/Entropy', dist_entropy.item(), global_train_step)
                global_train_step += 1  # 更新全局步数

    # 计算平均指标
    mean_policy_loss = total_policy_loss / batch_count if batch_count > 0 else 0
    mean_value_loss = total_value_loss / batch_count if batch_count > 0 else 0
    mean_entropy = total_entropy / batch_count if batch_count > 0 else 0

    print('Filtered {} transitions; Updated'.format(len(filter_index)))
    if return_metrics:
        return mean_value_loss, mean_policy_loss, mean_entropy, global_train_step
    else:
        return global_train_step  # 返回更新后的全局步数


def run(comm, env, policy, policy_path, action_bound, optimizer, writer):  # 新增writer参数
    rate = rospy.Rate(40)
    buff = []
    global_update = 0
    global_step = 0
    global_train_step = 0  # 新增全局训练步数

    if env.index == 0:
        env.reset_world()

    for id in range(MAX_EPISODES):
        env.reset_pose()
        env.generate_goal_point()
        group_terminal = False
        ep_reward = 0
        liveflag = True
        step = 1

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
                # get informtion
                r, terminal, result = env.get_reward_and_terminate(step)
                step += 1

            if liveflag == True:
                ep_reward += r
            if terminal == True:
                liveflag = False

            global_step += 1

            # get next state
            s_next = env.get_laser_observation()
            left = obs_stack.popleft()
            obs_stack.append(s_next)
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
            if env.index == 0:
                buff.append((state_list, a, r_list, terminal_list, logprob, v))
                if len(buff) > HORIZON - 1:
                    s_batch, goal_batch, speed_batch, a_batch, r_batch, d_batch, l_batch, v_batch = \
                        transform_buffer(buff=buff)
                    filter_index = get_filter_index(d_batch)
                    # print len(filter_index)
                    t_batch, advs_batch = generate_train_data(rewards=r_batch, gamma=GAMMA, values=v_batch,
                                                              last_value=last_v, dones=d_batch, lam=LAMDA)
                    memory = (s_batch, goal_batch, speed_batch, a_batch, l_batch, t_batch, v_batch, r_batch, advs_batch)
                    global_train_step = ppo_update_stage2(policy=policy, optimizer=optimizer, batch_size=BATCH_SIZE,
                                                          memory=memory, filter_index=filter_index,
                                                          epoch=EPOCH, coeff_entropy=COEFF_ENTROPY,
                                                          clip_value=CLIP_VALUE, num_step=HORIZON,
                                                          num_env=NUM_ENV, frames=LASER_HIST,
                                                          obs_size=OBS_SIZE, act_size=ACT_SIZE,
                                                          writer=writer, global_train_step=global_train_step)

                    buff = []
                    global_update += 1

            state = state_next

        if env.index == 0:
            # 新增：记录Episode奖励到TensorBoard
            if writer is not None:
                writer.add_scalar('Episode/Reward', ep_reward, global_update)

            if global_update != 0 and global_update % 20 == 0:
                torch.save(policy.state_dict(), policy_path + '/stage2_6_{}.pth'.format(global_update))
                logging.getLogger('mylogger').info('########################## model saved when update {} times#########'
                                                   '################'.format(global_update))

        logging.getLogger('mylogger').info('Env %02d, Goal (%05.1f, %05.1f), Episode %05d, setp %03d, Reward %-5.1f, %s,' % \
                                           (env.index, env.goal_point[0], env.goal_point[1], id, step - 1, ep_reward, result))
        logging.getLogger('loggercal').info(ep_reward)


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
    cal_f_handler.setLevel(logging.INFO)
    logger_cal.addHandler(cal_f_handler)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    env = StageWorld(512, index=rank, num_env=NUM_ENV)
    reward = None
    action_bound = [[-0.5, -1], [2, 1]]
    # torch.manual_seed(1)
    # np.random.seed(1)

    if rank == 0:
        policy_path = 'policy/policy_ori'
        # 假设原模型的 frames 为 old_frames，新的 frames 为 new_frames
        old_frames = 3
        new_frames = 6

        # 加载原模型
        old_model = CNNPolicy(old_frames, ACT_SIZE)
        # 这里可以加载原模型的参数，例如：
        # old_model.load_state_dict(torch.load('old_model.pth'))
        file = policy_path + '/stage2_2200.pth'
        if os.path.exists(file):
            logger.info('####################################')
            logger.info('############Loading Model###########')
            logger.info('####################################')
            state_dict = torch.load(file)
            old_model.load_state_dict(state_dict)
        else:
            logger.info('#####################################')
            logger.info('############Start Training###########')
            logger.info('#####################################')

        # 创建新模型
        new_model = CNNPolicy(new_frames, ACT_SIZE)

        # 复制原模型的参数到新模型
        for name, param in old_model.named_parameters():
            if 'act_fea_cv1.weight' in name or 'crt_fea_cv1.weight' in name:
                # 扩展卷积层的输入通道
                new_param = new_model.state_dict()[name]
                new_param[:, :old_frames, :] = param
                new_model.state_dict()[name].copy_(new_param)
            elif 'act_fea_cv1.bias' in name or 'crt_fea_cv1.bias' in name:
                new_model.state_dict()[name].copy_(param)
            elif 'act_fea_cv2' in name or 'act_fc1' in name or 'act_fc2' in name or 'actor1' in name or 'actor2' in name or 'crt_fea_cv2' in name or 'crt_fc1' in name or 'crt_fc2' in name or 'critic' in name:
                new_model.state_dict()[name].copy_(param)
        # policy = MLPPolicy(obs_size, act_size)
        policy = new_model

        policy.cuda()
        opt = Adam(policy.parameters(), lr=LEARNING_RATE)
        mse = nn.MSELoss()

        if not os.path.exists(policy_path):
            os.makedirs(policy_path)

       

        # 新增：创建TensorBoard写入器（仅Rank=0）
        log_dir = './tensorboard_logs/stage2'  # 日志保存路径
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
    else:
        policy = None
        policy_path = None
        opt = None
        writer = None  # 非Rank=0进程不写入

    try:
        # 新增：将writer传入run函数
        run(comm=comm, env=env, policy=policy, policy_path=policy_path,
            action_bound=action_bound, optimizer=opt, writer=writer)
    except KeyboardInterrupt:
        import traceback
        traceback.print_exc()
    finally:
        if rank == 0 and writer is not None:
            writer.close()  # 确保关闭写入器