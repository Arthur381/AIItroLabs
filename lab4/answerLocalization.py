from typing import List
import numpy as np
from utils import Particle
import math
### 可以在这里写下一些你需要的变量和函数 ###
COLLISION_DISTANCE = 1
MAX_ERROR = 50000
def edge(walls):
    x_min=np.min(walls,axis=0)[0]
    y_min=np.min(walls,axis=0)[1]
    x_max=np.max(walls,axis=0)[0]
    y_max=np.max(walls,axis=0)[1]
    return (x_min,x_max,y_min,y_max)

change_weight=0.50
guass_normal_pos=0.13
guass_normal_theta=np.pi/15
bias=0

    

### 可以在这里写下一些你需要的变量和函数 ###


def generate_uniform_particles(walls, N):#均匀采样，生成初始粒子
    """
    输入：
    walls: 维度为(xxx, 2)的np.array, 地图的墙壁信息，具体设定请看README关于地图的部分
    N: int, 采样点数量
    输出：
    particles: List[Particle], 返回在空地上均匀采样出的N个采样点的列表，每个点的权重都是1/N
    """
    all_particles: List[Particle] = []
    for _ in range(N):
        all_particles.append(Particle(1.0, 1.0, 1.0, 0.0))
    ### 你的代码 ###
    #第一次不需要高斯噪声
    #print(edge(walls))
    cur_edge=edge(walls)
    cnt=0
    while(cnt<N):
        x = np.random.uniform(cur_edge[0], cur_edge[1])
        y = np.random.uniform(cur_edge[2], cur_edge[3])
        #if np.array([x,y]) not in walls:
        '''if np.sum(np.all(np.isclose(walls, [x, y]), axis=1)) == 0:'''
        #any(np.linalg.norm(all_particles[i].position - wall) < COLLISION_DISTANCE for wall in walls):
        theta=np.random.uniform(0,2*np.pi)# theta 时弧度制
        all_particles[cnt]=Particle(x,y,theta,1.0/N)
        cnt+=1
    ### 你的代码 ###
    return all_particles


def calculate_particle_weight(estimated, gt):#计算该粒子的重采样权重
    """
    输入：
    estimated: np.array, 该采样点的距离传感器数据
    gt: np.array, Pacman实际位置的距离传感器数据
    输出：
    weight, float, 该采样点的权重
    """
    weight = 1.0
    ### 你的代码 ###
    #这里采用了 L2 距离
    weight=np.exp(-change_weight*np.linalg.norm(gt - estimated, ord=2))
    ### 你的代码 ###
    return weight


def resample_particles(walls, particles: List[Particle]):
    """
    输入：
    walls: 维度为(xxx, 2)的np.array, 地图的墙壁信息，具体设定请看README关于地图的部分
    particles: List[Particle], 上一次采样得到的粒子，注意是按权重从大到小排列的
    输出：
    particles: List[Particle], 返回重采样后的N个采样点的列表
    """
    resampled_particles: List[Particle] = []
    for _ in range(len(particles)):
        resampled_particles.append(Particle(1.0, 1.0, 1.0, 0.0))
    ### 你的代码 ###
    cur_edge=edge(walls)
    # 之前已经排好序且已经归一化
    #   我选择在这里添加噪声，即在采样的时候直接添加噪声
    lens=len(particles)
    cnt=0
    for i in range(lens):
        this_num=int(lens*particles[i].get_weight()*1.1)
        for k in range(this_num):
            x_noisy = np.random.normal(particles[i].position[0],guass_normal_pos)
            y_noisy = np.random.normal(particles[i].position[1],guass_normal_pos)
            #if np.array([x_noisy,y_noisy]) not in walls:
            theta_noisy = np.random.normal(particles[i].theta,guass_normal_theta)
            resampled_particles[cnt] = Particle(x_noisy,y_noisy,theta_noisy,1.0/lens)
            cnt+=1
            if cnt==lens:
                break
        if cnt==lens:
            break
    # Create a new particle with the noisy position
    cur_edge=edge(walls)
    #print("dis: ",lens-cnt)
    while cnt<lens:
        
        x = np.random.uniform(cur_edge[0], cur_edge[1])
        y = np.random.uniform(cur_edge[2], cur_edge[3])
        #if np.array([x,y]) not in walls:
        resampled_particles[cnt]=Particle(x,y,np.random.uniform(0,2*np.pi),1.0/lens)
        cnt+=1
    ### 你的代码 ###
    return resampled_particles

def apply_state_transition(p: Particle, traveled_distance, dtheta):#实现移动
    """
    输入：
    p: 采样的粒子
    traveled_distance, dtheta: ground truth的Pacman这一步相对于上一步运动方向改变了dtheta，并移动了traveled_distance的距离
    particle: 按照相同方式进行移动后的粒子
    """
    ### 你的代码 ###
    p.position[0]+=np.cos(p.theta)*traveled_distance
    p.position[1]+=np.sin(p.theta)*traveled_distance
    p.theta=(p.theta+dtheta)%(2*np.pi)
    ### 你的代码 ###
    return p

def get_estimate_result(particles: List[Particle]):#移动结束
    """
    输入：
    particles: List[Particle], 全部采样粒子
    输出：
    final_result: Particle, 最终的猜测结果
    """
    final_result = Particle()
    ### 你的代码 ###
    #之前已经由大到小排过了序
    final_result=particles[0]
    ### 你的代码 ###
    return final_result