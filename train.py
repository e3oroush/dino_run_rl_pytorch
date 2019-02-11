import torch
from models import QNetwork, Agent, GameState, build_keras_model
from utils import DinoSeleniumEnv,IMAGE_CHANNELS, IMAGE_WIDTH, IMAGE_HEIGHT, get_logger
import numpy as np
from collections import deque
import time


logger = get_logger("training", "training.log")
def train():
    chrome_driver_path = "/home/esoroush/github/chromedriver"
    checkpoint_path = "model.pth"
    nb_actions = 2
    epsilon = 0.1
    initial_epsilon = 0.1
    final_epsilon = 1e-4
    gamma = 0.99
    nb_memory = 5000
    nb_expolre = 100000
    is_debug = False
    batch_size = 32
    nb_observation = 1000
    seed = 22
    is_cuda = True
    np.random.seed(seed)
    memory = deque()
    env = DinoSeleniumEnv(chrome_driver_path)
    agent = Agent(env)
    game_state = GameState(agent, debug=is_debug)
    qnetwork = QNetwork(nb_actions)#build_keras_model()#QNetwork(nb_actions)
    if is_cuda:
        qnetwork.cuda()
    optimizer = torch.optim.Adam(qnetwork.parameters())
    try:
        m = torch.load(checkpoint_path)
        qnetwork.load_state_dict(m['qnetwork'])
        optimizer.load_state_dict(m['optimizer'])
        # qnetwork.load_weights("./model.h5")
    except:
        logger.warn("No model found in {}".format(checkpoint_path))
    loss_fcn = torch.nn.MSELoss()
    action_indx = 0 # do nothing as the first action
    screen, reward, is_gameover, score = game_state.get_state(action_indx)
    current_state = np.expand_dims(screen,0)
    current_state = np.tile(current_state,(IMAGE_CHANNELS,1,1)) # [IMAGE_CHANNELS,IMAGE_WIDTH,IMAGE_HEIGHT]
    initial_state = current_state

    t = 0
    last_time = 0
    sum_scores = 0
    total_loss = 0
    max_score = 0
    qvalues = np.array([0,0])
    try:
        while True:
            qnetwork.eval()
            if np.random.random() < epsilon: # epsilon greedy
                action_indx = np.random.randint(nb_actions)
            else:
                # tensor = np.expand_dims(current_state.transpose(1,2,0),0)
                # qvalues = qnetwork.predict(tensor).squeeze()
                # action_indx = np.argmax(qvalues)
                tensor = torch.from_numpy(current_state).float().unsqueeze(0)
                with torch.no_grad():
                    qvalues = qnetwork(tensor).squeeze()
                _,action_indx = qvalues.max(-1)
                qvalues = qvalues.cpu().numpy()
                action_indx = action_indx.item()
            if epsilon > final_epsilon and t > nb_observation:
                epsilon -= (initial_epsilon - final_epsilon) / nb_expolre 
            screen, reward, is_gameover, score = game_state.get_state(action_indx)
            sum_scores += score
            if score > max_score:
                max_score = score
            if last_time and t % 1000 == 0:
                logger.info('fps: {0}'.format(1 / (time.time()-last_time))) 
            last_time = time.time()
            screen = np.expand_dims(screen,0)
            next_state = np.append(screen,current_state[:IMAGE_CHANNELS-1,:,:], axis=0)
            memory.append((current_state, action_indx, reward, next_state, is_gameover))
            if len(memory) > nb_memory:
                memory.popleft()
            if t > nb_observation:
                indxes = np.random.choice(len(memory), batch_size, replace=False)
                minibatch = [memory[b] for b in indxes]
                inputs = np.zeros((batch_size,IMAGE_CHANNELS,IMAGE_WIDTH,IMAGE_HEIGHT))
                # inputs = np.zeros((batch_size,IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_CHANNELS))
                targets = np.zeros((batch_size,nb_actions))
                for i,(state_t,action_t,reward_t,state_t1,is_gameover_t1) in enumerate(minibatch):
                    # inputs[i] = state_t.transpose(1,2,0)
                    # tensor = np.expand_dims(state_t.transpose(1,2,0),0)
                    # qvalues = qnetwork.predict(tensor).squeeze()
                    # targets[i] = qvalues
                    inputs[i] = state_t
                    tensor = torch.from_numpy(state_t).float().unsqueeze(0)
                    with torch.no_grad():
                        qvalues = qnetwork(tensor).squeeze()
                    qvalues = qvalues.cpu().numpy()
                    targets[i] = qvalues
                    if is_gameover_t1:
                        assert reward_t == -1
                        targets[i,action_t] = reward_t
                    else:
                        tensor = torch.from_numpy(state_t1).float().unsqueeze(0)
                        with torch.no_grad():
                            qvalues = qnetwork(tensor).squeeze()
                            qvalues = qvalues.cpu().numpy()
                        # tensor = np.expand_dims(state_t1.transpose(1,2,0),0)
                        # qvalues = qnetwork.predict(tensor).squeeze()
                        targets[i,action_t] = reward_t + gamma * qvalues.max()
                qnetwork.train()
                qnetwork.zero_grad()
                inputs = torch.from_numpy(inputs).float()
                targets = torch.from_numpy(targets).float()
                if is_cuda:
                    targets = targets.cuda()
                q_values = qnetwork(inputs)
                loss = loss_fcn(q_values,targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                # total_loss += qnetwork.train_on_batch(inputs, targets)
            current_state = initial_state if is_gameover else next_state
            # env.restart_game()
            t+=1
            if t % 1000 == 0:
                env.pause_game()
                logger.info("For episode {}: mean score is {} max score is {} mean loss: {}".format(t, sum_scores/t, max_score, total_loss/t))
                logger.info("Episode: {} action_index: {} reward: {} max qvalue: {}".format(t,action_indx,reward,qvalues.max()))  
                # qnetwork.save_weights("model.h5")
                torch.save({"qnetwork":qnetwork.state_dict(),
                        "optimizer":optimizer.state_dict()},checkpoint_path)
                env.resume_game()
    except KeyboardInterrupt:
        # qnetwork.save_weights("model.h5")
        torch.save({"qnetwork":qnetwork.state_dict(),
                    "optimizer":optimizer.state_dict()},checkpoint_path)

if __name__ == '__main__':
    train()





