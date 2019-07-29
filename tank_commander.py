import tensorflow as tf
from tensorflow.keras import datasets, layers, models, preprocessing, callbacks
import numpy as np
import random


class TankCommander:
    """A model that commands tanks and learns"""
    
    def __init__(self, displacement=.2, epsilon=.1):
        self.history = []
        self.graph = self.build_graph()
        self.displacement = displacement
        self.n_exec_decision = 0
        if epsilon:
          self.epsilon = epsilon
        else:
          self.epsilon = None
        
        self.reward_graph = []
    
    
    def build_graph(self):
        inputs = layers.Input(shape=(44, 148, 3))
        x = layers.Conv2D(16, (3, 3), activation='relu')(inputs)
        x = layers.BatchNormalization(axis=-1)(x) # Channels @ -1
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Flatten()(x)
        x = layers.Dense(64)(x)
        x = layers.Activation('relu')(x)
        x = layers.Dense(64)(x)
        x = layers.Activation('sigmoid')(x)
        x = layers.BatchNormalization(axis=-1)(x)
        x = layers.Dropout(0.02)(x)
        x = layers.Dense(18, activation='linear')(x)
        model = models.Model(inputs, x)
        model.compile(optimizer='sgd',
              loss='mae',
              metrics=['accuracy'])
        return model
    
    def decision_function(self, obs=None, env=None):
      """Log every 10,000 steps"""
      if self.n_exec_decision % 10000 == 0:
          print(f"Predicting observation number {self.n_exec_decision}")
      self.n_exec_decision += 1
      
      # Periscope bounding box
      x, y, z = ((80, 124), (12, 160), (None))
      
      # Predictions and actions
      prediction = self.graph.predict(
          np.array([obs[x[0]:x[1], y[0]:y[1], :]])
      )
      
      # index of the highest scored action by our graph
      action = np.argmax(prediction)
      
      return action, prediction
    
    def train_on_history(self, history):
      """Organize the training jobs by each in-game life"""
      
      # Split into episodes
      n_episodes = history[-1]["episode"] 
      episodes = [list(filter(lambda h: h["episode"]==e , history)
                      ) for e in range(n_episodes)
      ]

      # Split into game lives
      for episode in episodes:
        
        
        game_lives = [
            list(filter(lambda h: h.get('info').get('ale.lives')==l, episode)
        ) for l in range(5)
        ]
        
        for life in game_lives:
            if life:
                self.train(life)
            else:
                print("No ocurrance")
      return
    
    def train(self, game_life):
      """Train on a single in-game life"""
      rewards = [obs.get('reward') for obs in game_life]
      cum_rewards = sum(rewards)

      # manage the graphics
      self.reward_graph.append(cum_rewards)
      plt.plot(self.reward_graph)
      x, y, z = peri_bounding_box

      # The amound of nudge
      if cum_rewards:
          displacement = cum_rewards * self.displacement
      else:
          displacement = 0 - self.displacement

      # Store observations and perturbed predictions
      data, targets = [], []

      for obs in game_life:

          # Perturb action
          action, prediction = obs.get('action')
          if self.epsilon and (random.uniform(0, 1.0) < self.epsilon):
            action = random.randrange(18)

          # Copy
          update = list(prediction)

          # Update only the target action
          update[0][action] = update[0][action] + displacement


          data.append(
              # Apply bounding box before appending
              np.array(obs.get('observation')[x[0]:x[1], y[0]:y[1], :])
          )


          update = np.array(update).reshape(1,18),
          targets.append(update)

      if data and len(data) > 15:
        # Image processing
        datagen = preprocessing.image.ImageDataGenerator(
          featurewise_center=True,
          featurewise_std_normalization=True,
          rotation_range=20,
          width_shift_range=0.2,
          height_shift_range=0.2,
          horizontal_flip=True)
        datagen.fit(data)

        # Training data generator
        train = datagen.flow(np.array(data), np.squeeze(np.array(targets)),
                             batch_size=16)

        # Finally train
        self.graph.fit_generator(train, steps_per_epoch=len(data)/16,
                                 epochs=30, verbose=0,
                                 callbacks=[
                                   callbacks.EarlyStopping(monitor='acc'),
                                   # callbacks.ModelCheckpoint() ?
                                  ]
                                 )
        
model = TankCommander()