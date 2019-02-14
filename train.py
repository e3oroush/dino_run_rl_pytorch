import torch
from models import QNetwork, Agent, GameState
from utils import DinoSeleniumEnv,IMAGE_CHANNELS, IMAGE_WIDTH, IMAGE_HEIGHT, get_logger
import numpy as np
from collections import deque
import time
import pickle
import gc
from argparse import ArgumentParser


logger = get_logger("training", "training.log")
def train(args):
    chrome_driver_path = args.chrome_driver_path
    checkpoint_path = args.checkpoint_path
    nb_actions = args.nb_actions
    initial_epsilon = args.initial_epsilon
    epsilon = initial_epsilon
    final_epsilon = args.final_epsilon
    gamma = args.gamma
    nb_memory = args.nb_memory
    nb_expolre = args.nb_expolre
    is_debug = args.is_debug
    batch_size = args.batch_size
    nb_observation = args.nb_observation
    desired_fps = args.desired_fps
    is_cuda = True if args.use_cuda and torch.cuda.is_available() else False
    log_frequency = args.log_frequency
    save_frequency = args.save_frequency
    ratio_of_win = args.ratio_of_win
    if args.exploiting:
        nb_observation = -1
        epsilon = final_epsilon

    seed = 22
    np.random.seed(seed)
    memory = deque()
    env = DinoSeleniumEnv(chrome_driver_path, speed=args.game_speed)
    agent = Agent(env)
    game_state = GameState(agent, debug=is_debug)
    qnetwork = QNetwork(nb_actions)
    if is_cuda:
        qnetwork.cuda()
    optimizer = torch.optim.Adam(qnetwork.parameters(),1e-4)
    tmp_param = next(qnetwork.parameters())
    try:
        m = torch.load(checkpoint_path)
        qnetwork.load_state_dict(m["qnetwork"])
        optimizer.load_state_dict(m["optimizer"])
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
    lost_action=[]
    win_actions=[]
    action_random=0
    action_greedy=0
    episodes = 0
    nb_episodes = 0
    if not args.exploiting:
        try:
            t,memory,epsilon,nb_episodes = pickle.load(open("cache.p", "rb"))
        except:
            logger.warn("Could not load cache file! Starting from scratch.")
    try:
        while True:
            qnetwork.eval()
            if np.random.random() < epsilon: # epsilon greedy
                action_indx = np.random.randint(nb_actions)
                action_random+=1
            else:
                action_greedy+=1
                tensor = torch.from_numpy(current_state).float().unsqueeze(0)
                with torch.no_grad():
                    qvalues = qnetwork(tensor).squeeze()
                _,action_indx = qvalues.max(-1)
                action_indx = action_indx.item()
            if epsilon > final_epsilon and t > nb_observation:
                epsilon -= (initial_epsilon - final_epsilon) / nb_expolre
            screen, reward, is_gameover, score = game_state.get_state(action_indx)
            if is_gameover:
                episodes+=1
                nb_episodes+=1
                lost_action.append(action_indx)
                sum_scores += score
            else:
                win_actions.append(action_indx)
            if score > max_score:
                max_score = score
            if last_time:
                fps = 1 / (time.time()-last_time)
                if fps > desired_fps:
                    time.sleep(1/desired_fps - 1/fps)
            if last_time and t % log_frequency == 0:
                logger.info('fps: {0}'.format(1 / (time.time()-last_time)))
            last_time = time.time()
            screen = np.expand_dims(screen,0)
            next_state = np.append(screen,current_state[:IMAGE_CHANNELS-1,:,:], axis=0)
            if not args.exploiting and (is_gameover or np.random.random() < ratio_of_win): # we store all lost action incomes and some of win actions!
                memory.append((current_state, action_indx, reward, next_state, is_gameover))
            if len(memory) > nb_memory:
                memory.popleft()
            if nb_observation > 0 and t > nb_observation:
                indxes = np.random.choice(len(memory), batch_size, replace=False)
                minibatch = [memory[b] for b in indxes]
                inputs = tmp_param.new(batch_size,IMAGE_CHANNELS,IMAGE_WIDTH,IMAGE_HEIGHT).zero_()
                targets =  tmp_param.new(batch_size,nb_actions).zero_()
                for i,(state_t,action_t,reward_t,state_t1,is_gameover_t1) in enumerate(minibatch):
                    inputs[i] = torch.from_numpy(state_t).float()
                    tensor = inputs[i].unsqueeze(0)
                    with torch.no_grad():
                        qvalues = qnetwork(tensor).squeeze()
                    targets[i] = qvalues
                    if is_gameover_t1:
                        assert reward_t == -1
                        targets[i,action_t] = reward_t
                    else:
                        tensor = torch.from_numpy(state_t1).float().unsqueeze(0)
                        with torch.no_grad():
                            qvalues = qnetwork(tensor).squeeze()
                        qvalues = qvalues.cpu().numpy()
                        targets[i,action_t] = reward_t + gamma * qvalues.max()
                qnetwork.train()
                qnetwork.zero_grad()
                q_values = qnetwork(inputs)
                loss = loss_fcn(q_values,targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            current_state = initial_state if is_gameover else next_state
            t+=1
            if t % log_frequency == 0:
                logger.info("For t {}: mean score is {} max score is {} mean loss: {} number of episode: {}".format(t, sum_scores/(episodes+0.1), max_score, total_loss/1000, episodes))
                logger.info("t: {} action_index: {} reward: {} max qvalue: {} total number of eposodes so far: {}".format(t,action_indx,reward,qvalues.max(), nb_episodes))
                tmp = np.array(lost_action)
                dnc = (tmp==0).sum()
                logger.info("Lost actions do_nothing: {} jump: {} length of memory {}".format(dnc, len(tmp)-dnc, len(memory)))
                tmp = np.array(win_actions)
                dnc = (tmp==0).sum()
                logger.info("Win actions do_nothing: {} jump: {}".format(dnc, len(tmp)-dnc))
                logger.info("Greedy action {} Random action {}".format(action_greedy, action_random))
                action_greedy=0
                action_random=0
                lost_action=[]
                win_actions=[]
                if episodes !=0:
                    sum_scores = 0
                total_loss = 0
                episodes = 0
            if t % save_frequency and not args.exploiting == 0:
                env.pause_game()
                with open("cache.p", "wb") as fh:
                    pickle.dump((t,memory,epsilon, nb_episodes), fh)
                gc.collect()
                torch.save({"qnetwork":qnetwork.state_dict(),
                        "optimizer":optimizer.state_dict()},checkpoint_path)
                env.resume_game()
    except KeyboardInterrupt:
        if not args.exploiting:
            torch.save({"qnetwork":qnetwork.state_dict(),
                        "optimizer":optimizer.state_dict()},checkpoint_path)
            with open("cache.p", "wb") as fh:
                pickle.dump((t,memory,epsilon, nb_episodes), fh)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--chrome_driver_path", help="Path of chrome driver", default="/home/esoroush/github/chromedriver")
    parser.add_argument("--checkpoint_path", help="Path of Pytorch model path", default="model.pth")
    parser.add_argument("--nb_actions", help="Number of possible actions for Dino.\
         Default is 2 but you can increase to 3 with do_nothing,jump and dive actions.", default=2, type=int)
    parser.add_argument("--initial_epsilon", help="Starting epsilon value for explorations", default=0.1, type=float)
    parser.add_argument("--final_epsilon", help="Final value for epsilon after exploration", default=1e-4, type=float)
    parser.add_argument("--gamma", help="Value of gamma for atenuation of rewards in next states.", default=0.99, type=float)
    parser.add_argument("--nb_memory", help="Number of memory to store previous states and rewards for training.", default=50000, type=int)
    parser.add_argument("--nb_expolre", help="Number of times for explorations.\
         After this time the epsilon is in final_epsilon value and the explorations is in its minumum value.", default=100000, type=int)
    parser.add_argument("--is_debug", help="A flag for debugging.\
         If enabled an OpenCV window is shown that illustrates the feeded image to the netwrok", action="store_true")
    parser.add_argument("--batch_size", help="Number of batch for training.", default=16, type=int)
    parser.add_argument("--nb_observation", help="Number of observation before starting training", default=100, type=int)
    parser.add_argument("--use_cuda", help="Use cuda if it's avaulable", dest="use_cuda", action="store_true")
    parser.add_argument("--exploiting", help="If enabled, there is no training the qnetwork just predict the qvalues.", action="store_true")
    parser.add_argument("--log_frequency", help="Frequency of logging every time step.", default=1000, type=int)
    parser.add_argument("--save_frequency", help="Frequency of saving satate of the training", default=10000, type=int)
    parser.add_argument("--game_speed", help="Speed of the game.", default=0, type=int)
    parser.add_argument("--ratio_of_win", help="Ration of usage of win actions in training. It should be between (0,1].\
         1 means use all actions and 1e-6 means small amount of win actions.", default=1., type=float)
    parser.add_argument("--desired_fps", help="If you want to reduce processing fps in order to have constant fps in training and testing time.",\
             default=25, type=int)
    parser.set_defaults(use_cuda=True)
    args = parser.parse_args()
    train(args)





