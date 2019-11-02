#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style


# In[ ]:


style.use("ggplot")

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
fig.canvas.set_window_title('Live Graph')




# In[ ]:


def animate(i):

    
    
    pull_data = open("twitter-out.txt", "r").read()
    lines = pull_data.split('\n')

    xar = []
    yar = []

    x = 0
    y = 0

    for l in lines[-200:]:
        x += 1
        if "pos" in l:
            y += 1
        elif "neg" in l:
            y -= 1

        xar.append(x)
        yar.append(y)

    ax1.clear()
    plt.title('Sentiment Analysis: Live Graph')
    plt.xlabel('Number of tweets')
    plt.ylabel('Sentiment score')
    ax1.plot(xar, yar)


# In[ ]:


ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




