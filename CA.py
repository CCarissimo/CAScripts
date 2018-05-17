import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from matplotlib.animation import FuncAnimation
# import random
# import time
# import scipy 
from scipy import ndimage
from PIL import Image
from model import Counter
from model import Rule
from model import CA2D
from model import CA2D_boundary_free
from model import import_image
from model import Counter2
from model import Rule2
from alphabet import alphabet
from model import words
from model import create_canvas
from model import insert_words
from model import collage
from PIL import Image

    #### Here create the initial input array ####

#m = np.array([[[1 for x in range(50)] for y in range(50)] for z in range(2)])

#m = np.array([[[0 if x%2==0 else 1 for x in range(50)] for y in range(50)] for z in range(2)])
#m[-1, -1, -1] = 1
# love = words('l o v e', alphabet)


# rotation = words('&', alphabet)


# #m[-1, 53:56, 43:58] = love
# #m[-1, 48:52, 35:65] = rotation
# new_dionysus = import_image('Dionysus.jpg', 110)

# m[-1, :, :] = new_dionysus

m = create_canvas(500, 500)
circles = import_image('arrivals.jpg', 150)
m[-1, :, :] = circles

plt.imshow(circles)
plt.show()

text = words('T H I S  I S  N O T  Y O U R  T Y P I C A L  S E T T I N G  ', alphabet)
text1 = words('T H I S  I S  N O T  Y O U R  T Y P I C A L  S E T T I N G  U N L E S S  Y O U  ', alphabet)
text2 = words('T H I S  I S  N O T  Y O U R  T Y P I C A L  S E T T I N G  U N L E S S  Y O U  W A N T  ', alphabet)
text3 = words('T H I S  I S  N O T  Y O U R  T Y P I C A L  S E T T I N G  U N L E S S  Y O U  W A N T  T O  B E  L O S T ', alphabet)
text4 = words('T H I S  I S  N O T  Y O U R  T Y P I C A L  S E T T I N G  U N L E S S  Y O U  W A N T  T O  B E  L O S T  in translation  we  trust  ', alphabet)

# texts = [text, text1, text2, text3, text4]
# canvas = insert_words(text4, m)
# iterations_per_image = [20, 15, 4, 10, 2]

full_CellularAutomata = CA2D_boundary_free(m, 50)

fig = plt.figure()

CA = plt.imshow(full_CellularAutomata[0], vmin=0, vmax=1, animated=True, cmap="binary")

def init():
    CA.set_data(full_CellularAutomata[0])
    return CA

def animate(i):
	newIteration = full_CellularAutomata[i]
	#print newIteration
	CA.set_data(newIteration)
	return CA

anim = FuncAnimation(fig, animate, init_func=init, frames=(len(full_CellularAutomata[:,0,0])), interval=200)

plt.axis("off")
plt.title("It's a love thing")
#anim.save("arrivals.mp4")
plt.show()