from string import ascii_uppercase
from draw_utils import *
from pyglet.gl import *
import numpy as np
import pandas as pd
import os
import torch

# reward
move_reward = -0.1
obs_reward = -0.5
goal_reward = 10.0
print('reward:' , move_reward, obs_reward, goal_reward)

local_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))

class Simulator:
    def __init__(self,test = False):
        '''
        height : 그리드 높이
        width : 그리드 너비 
        inds : A ~ Q alphabet list
        '''
        # Load train data
        self.files = pd.read_csv(os.path.join(local_path, "./data/factory_order_train.csv"))
# 테스트 파일 변수
        self.test_files = pd.read_csv(os.path.join(local_path, "./data/factory_order_test.csv"))
        self.test = False
        self.height = 10
        self.width = 9
        self.inds = list(ascii_uppercase)[:17]
# 아이템 박스에 진입할 수 있는 방향 설정
# (박스 좌표): (들어가는 방향, 나가는 방향)
        self.shelf = {(9, 4): (1,0), (5, 0): (2,3), (4, 0): (2,3), (3, 0): (2,3), (2, 0): (2,3),
                      (0, 0): (0,1), (0, 1): (0,1), (0, 2): (0,1), (0, 3): (0,1), (0, 4): (0,1),
                      (0, 5): (0,1), (0, 6): (0,1), (0, 7): (0,1), (0, 8): (0,1),
                      (2, 8): (3,2), (3, 8): (3,2), (4, 8): (3,2), (5, 8): (3,2)}

    def set_box(self):
        '''
        아이템들이 있을 위치를 미리 정해놓고 그 위치 좌표들에 아이템이 들어올 수 있으므로 그리드에 2으로 표시한다.
        데이터 파일에서 이번 에피소드 아이템 정보를 받아 가져와야 할 아이템이 있는 좌표만 3으로 표시한다.
        self.local_target에 에이전트가 이번에 방문해야할 좌표들을 저장한다.
        따라서 가져와야하는 아이템 좌표와 end point 좌표(처음 시작했던 좌표로 돌아와야하므로)가 들어가게 된다.

        수정)
        원핫인코딩을 위해 각 그리드를 0~5로 표시
        모든 타겟을 한번에 표시하지 않고 첫 타겟만 4로 표시
        나머지 아이템은 3, 아이템이 없는 곳은 2로 표시
        '''
        box_data = pd.read_csv(os.path.join(local_path, "./data/box.csv"))

        # 물건이 들어있을 수 있는 경우
        for box in box_data.itertuples(index = True, name ='Pandas'):
            self.grid[getattr(box, "row")][getattr(box, "col")] = 2

        # 물건이 실제 들어있는 경우
        order_item = list(set(self.inds) & set(self.items))
        order_csv = box_data[box_data['item'].isin(order_item)]

        for order_box in order_csv.itertuples(index = True, name ='Pandas'):
            self.grid[getattr(order_box, "row")][getattr(order_box, "col")] = 3
            # local target에 가야 할 위치 좌표 넣기
            self.local_target.append(
                [getattr(order_box, "row"),
                 getattr(order_box, "col")]
                )

# 좌표 기준으로 sort하면 안됨 ㅠㅠ
        #self.local_target.sort()
        self.local_target.append([9,4]) 
        self.origin_target = self.local_target.copy()

        # 알파벳을 Grid에 넣어서 -> grid에 2Dconv 적용 가능

    def set_obstacle(self):
        '''
        장애물이 있어야하는 위치는 미리 obstacles.csv에 정의되어 있다. 이 좌표들을 0으로 표시한다.
        
        수정)
        장애물의 방문횟수는 100으로 설정
        '''
        obstacles_data = pd.read_csv(os.path.join(local_path, "./data/obstacles.csv"))
        for obstacle in obstacles_data.itertuples(index = True, name ='Pandas'):
            self.grid[getattr(obstacle, "row")][getattr(obstacle, "col")] = 0
            #self.table[getattr(obstacle, "row")][getattr(obstacle, "col")] = 100


    def reset(self, ep):
        '''
        reset()은 첫 스텝에서 사용되며 그리드에서 에이전트 위치가 start point에 있게 한다.
        :param epi: episode, 에피소드 마다 가져와야 할 아이템 리스트를 불러올 때 사용
        :return: 초기셋팅 된 그리드
        :rtype: tensor(float32)
        _____________________________________________________________________________________
        items : 이번 에피소드에서 가져와야하는 아이템들
        terminal_location : 현재 에이전트가 찾아가야하는 목적지
        local_target : 한 에피소드에서 찾아가야하는 아이템 좌표, 마지막 엔드 포인트 등의 위치좌표들
        actions: visualization을 위해 에이전트 action을 저장하는 리스트
        curloc : 현재 위치
        '''

        # initial episode parameter setting
        self.ep = ep
        self.epi = ep%39999
        if self.test:
            self.items = list(self.test_files.iloc[self.epi])[0]
        else:
            self.items = list(self.files.iloc[self.epi])[0]
        self.cumulative_reward = 0
        self.terminal_location = None
        self.local_target = []
        self.actions = []

        # initial grid setting
        self.grid = np.ones((self.height, self.width), dtype="float16")
        
# 방문 횟수 테이블 세팅
        self.table = np.zeros((self.height, self.width), dtype="float16")

# -1로 세팅해서 처음 방문할 때 +0.1의 보상을 받게 함 (취소)
# 두번째 방문시 보상은 0, 세번째 방문시 보상은 -0.1 ... (취소)
        #self.table = np.ones((self.height, self.width), dtype="float16")*(-1)

        # set information about the gridworld
        self.set_box()
        self.set_obstacle()

        # start point를 grid에 표시
        self.curloc = [9, 4]
        self.grid[int(self.curloc[0])][int(self.curloc[1])] = 5
        
        self.done = False
        
# array(10,9)가 아니라 tensor(1,8,10,9) 상태 반환
        return self.get_state()

    def apply_action(self, action, cur_x, cur_y):
        '''
        에이전트가 행한 action대로 현 에이전트의 위치좌표를 바꾼다.
        action은 discrete하며 4가지 up,down,left,right으로 정의된다.
        
        :param x: 에이전트의 현재 x 좌표
        :param y: 에이전트의 현재 y 좌표
        :return: action에 따라 변한 에이전트의 x 좌표, y 좌표
        :rtype: int, int
        '''
        new_x = cur_x
        new_y = cur_y
        # up
        if action == 0:
            new_x = cur_x - 1
        # down
        elif action == 1:
            new_x = cur_x + 1
        # left
        elif action == 2:
            new_y = cur_y - 1
        # right
        else:
            new_y = cur_y + 1

        return int(new_x), int(new_y)


    def get_reward(self, new_x, new_y, out_of_boundary):
        '''
        get_reward함수는 리워드를 계산하는 함수이며, 상황에 따라 에이전트가 action을 옳게 했는지 판단하는 지표가 된다.
        :param new_x: action에 따른 에이전트 새로운 위치좌표 x
        :param new_y: action에 따른 에이전트 새로운 위치좌표 y
        :param out_of_boundary: 에이전트 위치가 그리드 밖이 되지 않도록 제한
        :return: action에 따른 리워드
        :rtype: float
        
        수정)
        각 그리드에서 방문횟수/10 만큼 패널티 적용
        '''

        # 바깥으로 나가는 경우
        if any(out_of_boundary):
            reward = obs_reward
                       
        else:
            # 장애물에 부딪히는 경우
# 빈 아이템 박스에 접근하는 경우
            if self.grid[new_x][new_y] in (0,2,3):
                reward = obs_reward  

            # 현재 목표에 도달한 경우
            elif self.grid[new_x][new_y] == 4:
                reward = goal_reward
# 출발지로 돌아오는경우 추가 보상
                if [new_x, new_y] == [9,4]:
                    reward += goal_reward

            # 그냥 움직이는 경우 
            else:
                #reward = move_reward
                reward = -(self.table[new_x][new_y]/10)

        return reward

    def step(self, action):
        ''' 
        에이전트의 action에 따라 step을 진행한다.
        action에 따라 에이전트 위치를 변환하고, action에 대해 리워드를 받고, 어느 상황에 에피소드가 종료되어야 하는지 등을 판단한다.
        에이전트가 endpoint에 도착하면 gif로 에피소드에서 에이전트의 행동이 저장된다.
        :param action: 에이전트 행동
        :return: # 리턴 수정
            s, 그리드(텐서)
            action, 에이전트 행동
            reward, 리워드
            cumulative_reward, 누적 리워드
            s_prime, 다음 상태(텐서)
            done, 종료 여부
            #goal_ob_reward, goal까지 아이템을 모두 가지고 돌아오는 finish율 계산을 위한 파라미터
        :rtype: tensor(float32), int, float, float, tensor(float32), bool, bool/str
        (Hint : 시작 위치 (9,4)에서 up말고 다른 action은 전부 장애물이므로 action을 고정하는 것이 좋음)

        수정)
        목표물에 도달했을 경우 반드시 후진으로 빠져나오게 함
        goal_ob_reward, 
            평소에는 False
            목표 아이템에 도달했을 때 True
            아이템을 다 먹고 도착지에 도달했을 때 'finish'
        '''

        self.terminal_location = self.local_target[0]
# 현재 목표를 4로 표시
        self.grid[int(self.terminal_location[0])][int(self.terminal_location[1])] = 4
        s = self.get_state()
        #print('start:', self.curloc, 'item:', self.terminal_location)
        #pprint(self.grid)
        cur_x,cur_y = self.curloc
        self.actions.append((cur_x, cur_y))
        #print('now:', self.curloc, 'action:', action)

        goal_ob_reward = False
        
        new_x, new_y = self.apply_action(action, cur_x, cur_y)

        out_of_boundary = [new_x < 0, new_x >= self.height, new_y < 0, new_y >= self.width]
# 허용되지 않은 방향에서 아이템 박스에 진입했을 경우 바깥으로 나간 것으로 판단
        if (new_x, new_y) in self.shelf:
            if action != self.shelf[(new_x, new_y)][0]:
                print('벽으로 못 들어감')
                out_of_boundary.append(True)
# 아이템 박스에서 허용되지 않은 방향으로 나가는 경우 바깥으로 나간 것으로 판단
        if (cur_x,cur_y) in self.shelf:
            if action != self.shelf[(cur_x,cur_y)][1]:
                print('벽으로 못 나감')
                out_of_boundary.append(True)
        
        # 바깥으로 나가는 경우 종료
        if any(out_of_boundary):
            print('아이쿠!')
            self.done = True
        else:
            # 장애물에 부딪히는 경우 종료
# 빈 아이템 박스에 접근하는 경우에도 종료
            if self.grid[new_x][new_y] in (0,2,3):
                print('아이쿠!')
                self.done = True

            # 현재 목표에 도달한 경우, 다음 목표설정
            elif self.grid[new_x][new_y] == 4:
                print('성공!')
                # end point 일 때
                if (new_x, new_y) == (9,4):
                    print('도착')
                    self.table = np.zeros((self.height, self.width), dtype="float16")
                    self.done = True
                else:
                    self.table[new_x][new_y] += 1
                    self.done = False
                    
                self.local_target.remove(self.local_target[0])
                #print(self.local_target)
                #input()
                self.grid[cur_x][cur_y] = 1
                self.grid[new_x][new_y] = 5
                goal_ob_reward = True
                self.curloc = [new_x, new_y]
                
            else:
# 아이템 박스에서 나가는 경우
                if (cur_x,cur_y) in self.shelf:
                    if (cur_x,cur_y) == (9,4):
                        self.grid[cur_x][cur_y] = 3
                    else:
                        self.grid[cur_x][cur_y] = 2
                    self.grid[new_x][new_y] = 5
                    self.table[new_x][new_y] += 1
                    self.curloc = [new_x,new_y]
                    self.done = False
                    print('나가자!')
                    
                # 그냥 길에서 움직이는 경우 
                else: 
                    self.grid[cur_x][cur_y] = 1
                    self.grid[new_x][new_y] = 5
                    self.table[new_x][new_y] += 1
                    self.curloc = [new_x,new_y]
                    self.done = False
                    print('..')

        #print(self.grid)
        #print(self.actions)
        #print('done:', self.done)
        #print()
                
        reward = self.get_reward(new_x, new_y, out_of_boundary)
        self.cumulative_reward += reward
        s_prime = self.get_state()
        
        #print(s_prime)

        if self.done == True:
# 완료되면 방문횟수 테이블 출력
            print(self.table)
            #print(s_prime)
            #input()
            
            if [new_x, new_y] == [9, 4]:
                if self.terminal_location == [9, 4]:
                    goal_ob_reward = 'finish'
# 학습중에는 GIF 저장 x, 테스트 파일 확인 할 때만 저장
                if self.test:
                    height = 10
                    width = 9 
                    display = Display(visible=False, size=(width, height))
                    display.start()
                    start_point = (9, 4)
                    unit = 50
                    screen_height = height * unit
                    screen_width = width * unit
                    log_path = "./logs"
                    data_path = "./data"
                    render_cls = Render(self.origin_target, screen_width, screen_height, unit, start_point, data_path, log_path)
                    for idx, new_pos in enumerate(self.actions):
                        render_cls.update_movement(new_pos, idx+1)
                    
                    render_cls.save_gif('_train', self.ep, self.actions)
                    render_cls.viewer.close()
                    display.stop()
        
        return s, action, reward, self.cumulative_reward, s_prime, self.done, goal_ob_reward
        
# 현재의 grid와 table을 하나의 텐서로 변경해주는 함수
    def get_state(self): 
        origin_grid = torch.from_numpy(self.grid.astype('int')) # tensor.size(10,9)
        origin_table = torch.from_numpy(self.table.astype('int')) # tensor.size(10,9)
        oh_state = torch.from_numpy(np.eye(6)[origin_grid]) # tensor.size(10,9,6)
        oh_state = torch.permute(oh_state, (2,0,1)) # tensor.size(6,10,9)
        new_state = torch.stack([origin_grid,origin_table]) # tensor.size(2,10,9)
        new_state = torch.concat([oh_state, new_state], dim=0) # tensor.size(8,10,9)
        new_state = torch.unsqueeze(new_state,0) # tensor.size(1,8,10,9)
        new_state = new_state.type(torch.float32)
        
        return new_state

if __name__ == "__main__":

    sim = Simulator()
    files = pd.read_csv("./data/factory_order_train.csv")

    for epi in range(1): # len(files)):
        items = list(files.iloc[epi])[0]
        done = False
        i = 0
        obs = sim.reset(epi)
        print(sim.local_target)
        #actions = [0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1]
        #actions = [0,0,2,0,0,0,0,0,0,0,0,3,3,3,3,3,1,2,2,2,1,1,1,1,1,1,1,2,1]
        actions = [0,0,0]

        while done == False:

            i += 1
            next_obs, action, reward, cumul, s_prime, done, goal_reward = sim.step(actions[i])

            obs = next_obs

            if (done == True) or (i == (len(actions)-1)):
                i =0

