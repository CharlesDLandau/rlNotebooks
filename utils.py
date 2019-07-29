import gym
from gym import wrappers
import io
import base64
from IPython.display import HTML

def run_job(env_name, model, steps, episodes_per_train=1,
            verbose=False, video_dir=False):
    env = gym.make(env_name)
    
    # Video output
    if not video_dir:
        video_dir = "./gym-results"
    
    env = wrappers.Monitor(env, video_dir, force=True)
    
    # initial observation & reset
    observation = env.reset()
    
    history = []
    
    episode = 0
    
    for i in range(steps):
        action = model.decision_function(obs=observation, env=env)
        if isinstance(action, int):
          observation, reward, done, info = env.step(action)
        else:
          observation, reward, done, info = env.step(action[0])
          
        history.append({
            "episode": episode,
            "observation": observation,
            "reward": reward,
            "done": done,
            "info": info,
            "action": action,
            "step": i
        })
        if done:
            if verbose:
                print(f"Done with episode {episode}, {info}")
            
            # Train model
            if episodes_per_train and episode % episodes_per_train==0:
                model.train_on_history(history)
            
            
            episode += 1
            env.reset()
    env.close()
    data = {
        'history': history,
        'env':env,
        'parameters': {
            'steps': steps,
            'env_name': env_name,
            'episodes_per_train': episodes_per_train,
            'video_dir': video_dir
        }
        
    }
    return data

#result = run_job("SpaceInvaders-v0", model,
#        1000, episodes_per_train=0);

def render_video(episode_num, env, video_dir=None):
    if not video_dir:
        video_dir = './gym-results'
    video_file_handle = '{}/openaigym.video.{}.video{:0>6}.mp4'.format(video_dir, env.file_infix, episode_num)
    with io.open(video_file_handle, 'r+b') as file:
        video = file.read()
        encoded = base64.b64encode(video)

        return HTML(data='''
        <video width="360" height="auto" alt="test" controls><source src="data:video/mp4;base64,{0}" type="video/mp4" /></video>'''
    .format(encoded.decode('ascii')))
    
# render_video(0, result['env']);