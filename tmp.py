import matplotlib.pyplot as plt
import numpy as np

r = np.linspace(0,1,200)

bluex1 = r * np.cos(2*np.pi*r)
bluey1 = r * np.sin(2*np.pi*r)

bluex2 = r * np.cos(2*np.pi*r+np.pi)
bluey2 = r * np.sin(2*np.pi*r+np.pi)

bluex = np.append(bluex1,bluex2)
bluey = np.append(bluey1,bluey2)

redx1 = r * np.cos(2*np.pi*r+(np.pi/2))
redy1 = r * np.sin(2*np.pi*r+(np.pi/2))

redx2 = r * np.cos(2*np.pi*r+(3*np.pi/2))
redy2 = r * np.sin(2*np.pi*r+(3*np.pi/2))

redx = np.append(redx1,redx2)
redy = np.append(redy1,redy2)

bluenoisex = np.random.normal(0,0.02,400)
bluenoisey = np.random.normal(0,0.02,400)
rednoisex = np.random.normal(0,0.02,400)
rednoisey = np.random.normal(0,0.02,400)

bluex = bluex + bluenoisex
bluey = bluey + bluenoisey

redx = redx + rednoisex
redy = redy + rednoisey

plt.figure(figsize=(5,8), dpi=300)
plt.ylim([-1.1,2])
plt.scatter(bluex, bluey, c='#505050', marker='.')
plt.scatter(redx, redy, c='#00A6D6', marker='.')
plt.axis('off')
plt.show()