import tensorflow as tf
import numpy as np

def threegate(chosen_gate, ifchange):
    car = np.random.randint(3)
    if((car==chosen_gate)and(ifchange==1))or((car!=chosen_gate)and(ifchange==0)):
        return -1
    else:
        return 1

weights = tf.Variable(tf.ones([2]))
chosen_action = tf.argmax(weights,0)

reward_holder = tf.placeholder(shape=[1],dtype=tf.float32)
action_holder = tf.placeholder(shape=[1],dtype=tf.int32)

responsible_weight = tf.slice(weights,action_holder,[1])

loss = -(tf.log(responsible_weight)*reward_holder)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
update = optimizer.minimize(loss)

total_episodes = 1000
total_reward = np.zeros(2)
e = 0.03 

init = tf.global_variables_initializer()

whetherchange = 0

with tf.Session() as sess:
    sess.run(init)
    i = 0
    while i < total_episodes:
        if np.random.rand(1) < e:
            action = np.random.randint(2)
        else:
            action = sess.run(chosen_action)

        whetherchange += action
        reward = threegate(1, action)

        _, ww = sess.run([update, weights], feed_dict={reward_holder:[reward],
                                                       action_holder:[action]})

        total_reward[action] += reward
        i+=1

print 'total rewards:', total_reward
print 'times change:', whetherchange
