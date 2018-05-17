# Python script with all of the Functions to run the CA model

import numpy as np
import matplotlib.pyplot as plt
import random
import scipy 

def Counter(cells):
    ones = sum([sum(cells[0, :]), sum(cells[1, :]), sum(cells[2, :])])
    if cells[1, 1] == 1:
        ones -= 1
    return(ones)

def Counter2(cells):
	target = 0
	ones = sum([sum(cells[0, :]), sum(cells[1, :]), sum(cells[2, :])])
	if cells[1, 1] == 1:
		ones -= 1
		target += 1
	return(ones, target)

def Rule(ones):
    if ones == 0:
        return 0
    elif ones < 2:
        return 0
    elif ones <= 5:
        return 1
    elif ones > 5:
        return 0
    elif ones == 8:
        return 0

def Rule2(ones, target):
	if target == 0:
	    if ones == 0:
	        return 0
	    elif ones < 3:
	        return 0
	    elif ones == 3:
	        return 1
	    elif ones > 3:
	        return 0
	else:
		if ones == 0:
			return 0
		elif ones < 2:
			return 0
		elif ones <= 3:
			return 1
		elif ones > 3:
			return 0

def CA2D(initial, iterations):
    # Creation of a master array of 0's with the necessary dimensions
    x_len = len(initial[0, 0,:])
    y_len = len(initial[0, :,])
    array = np.zeros((iterations, x_len, y_len))
    master = np.concatenate((initial, array), axis = 0)
    
    # Loop within which the array will be updated
    for y in range(1, iterations): 
        for i in range(1, x_len - 1):
            for j in range(1, y_len - 1):
                cells = np.array([[master[y, i + 1, j - 1], master[y, i + 1, j], master[y, i + 1, j + 1]], 
                         [master[y, i, j - 1], master[y, i, j], master[y, i, j + 1]], 
                        [master[y, i - 1, j - 1], master[y, i - 1, j], master[y, i - 1, j + 1]]])
                master[y + 1, i, j] = Rule(Counter(cells))
    return(master)

def CA2D_boundary_free(initial, iterations):
    # Creation of a master array of 0's with the necessary dimensions
    x_len = len(initial[0, 0,:])
    y_len = len(initial[0, :,])
    array = np.zeros((iterations, x_len, y_len))
    master = np.concatenate((initial, array), axis = 0)
    
    # Loop within which the array will be updated
    for y in range(1, iterations):
        
        left_edge = master[y, :, 0] # Save all the edges of the CA 
        right_edge = master[y, :, -1]
        top_edge = master[y, 0, :]
        bottom_edge = master[y, -1, :]

        topleft_corner = master[y, 0, 0] # Save all the corners of the CA
        topright_corner = master[y, 0, -1]
        bottomleft_corner = master[y, -1, 0]
        bottomright_corner = master[y, -1, -1]

        # Combine edges and corners to create two large edges and two smaller edges
        top_edge = np.lib.pad(top_edge, (1,1), "constant", constant_values=(bottomright_corner, bottomleft_corner))
        bottom_edge = np.lib.pad(bottom_edge, (1,1), "constant", constant_values=(topright_corner, topleft_corner))

        # Now add the edges and corners of the CA to a new matrix such that we extend the surface of the CA
        surface = np.zeros(((x_len + 2), (y_len + 2)))
        surface[0] = top_edge 
        surface[-1] = bottom_edge
        surface[1:-1, 0] = left_edge
        surface[1:-1, -1] = right_edge
        surface[1:-1, 1:-1] = master[y]
        
        for i in range(1, (len(surface[:,0]) - 1)):
            for j in range(1, (len(surface[0,:]) - 1)):
                cells = np.array([[surface[i + 1, j - 1], surface[i + 1, j], surface[i + 1, j + 1]], 
                         [surface[i, j - 1], surface[i, j], surface[i, j + 1]], 
                        [surface[i - 1, j - 1], surface[i - 1, j], surface[i - 1, j + 1]]])
                #master[y + 1, (i - 1), (j - 1)] = Rule2(Counter2(cells)[0], Counter2(cells)[1])
                master[y + 1, (i - 1), (j - 1)] = Rule(Counter(cells))
                
    return(master)

#Python script used to convert an image into black and white 

def import_image(image_name, black_white_balance):
	new_image = scipy.ndimage.imread("%s"% image_name, flatten=True)

	for i in range(len(new_image[:,0])):
	    for j in range(len(new_image[0, :])):
	        if new_image[i, j] >= black_white_balance:
	            new_image[i, j] = 0
	        else:
	            new_image[i, j] = 1
	return(new_image)



# Python function to create a word in array form from a dictionary 
def words(string_input, dictionary):
	word = string_input

	first_letter = dictionary[word[0]]

	for i in range(1, len(word)):
		first_letter = np.concatenate((first_letter, dictionary[word[i]]), axis=1)

	return(first_letter)

def create_canvas(x_size, y_size):
	canvas = np.array([[[0 for x in range(x_size)] for y in range(y_size)] for z in range(2)])
	return(canvas)


def insert_words(word_array, canvas):

	len_text = len(word_array[0,:])
	len_canvas = len(canvas[0,:, 0])
	
	if len_text >= len(canvas[0,:]) - 5:
		factor = len_text // len_canvas

		word_array = np.split(word_array, [(len_canvas - 5)*x for x in range(1, (factor + 1))], axis=1)

		init_pos = len_canvas // 2 - int((factor * 5)/2)
		for i in range(factor + 1):
			
			word = word_array[i]

			canvas[1, (init_pos+i*5):(init_pos+3+i*5), 0:(len(word[0,:]))] = word
	else:
		canvas[1,(2):(5), 2:(len_canvas-2)] = word_array

	return(canvas)

def collage(initial_canvases, iterations_per_image):
	full_CA = initial_canvases[0]
	initial_canvases = np.array(initial_canvases)
	for i in range(len(iterations_per_image)):
		CA = CA2D_boundary_free(initial_canvases[i,:,:,:], iterations_per_image[i])
		inv_CA = [CA[-i] for i in range(1, (len(CA[:,0,0])+1))]
		CA = np.concatenate((CA, inv_CA))
		full_CA = np.concatenate((full_CA, CA))

	return(full_CA)