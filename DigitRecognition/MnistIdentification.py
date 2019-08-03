# Importing nessecary modules/libraries
import keras
import matplotlib.pyplot as plt
import numpy as np
import os
import pygame
from PIL import Image
import cv2
import image
import time
import PIL
# Keras imports for setting up layers in neural network
from keras.models import Sequential
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten

# Create filepath for saving and boolean for whether or not to load the model
path = '/home/idstudent/Desktop/LukeM/testingdigits/MnistIdentification/saved_models/'
load_checkpoint = True
testing_mode = False
save_checkpoint = True



# Initializing the Keras model and adding the layers
model = Sequential()
model.add(Conv2D(64, (3, 3), activation = 'relu', input_shape = (28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Flatten())
model.add(Dense(128))
# Disables the paths
model.add(Dropout(rate = 0.3))
model.add(Dense(10, activation = 'softmax'))
# Summary of all the models layers/parameters
if load_checkpoint:
    model.load_weights(path + "mnist-cnn")

model.summary()

model.compile(loss = keras.losses.categorical_crossentropy, metrics = ["accuracy"], optimizer = keras.optimizers.Adadelta())

# Importing the MNIST dataset from the Keras library and then loads it
if not testing_mode:
    from keras.datasets import mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshapes the data into readable/processable information for the network
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    # One Hot Encoding to train and test the labels
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)

    # TRAINING
    if not testing_mode:
        model.fit(x_train, y_train, epochs=2, batch_size=64)

    # Saving the model
    if not os.path.exists(path):
        os.makedirs(path)
    if save_checkpoint:
        model.save(path + 'mnist-cnn', overwrite=True, include_optimizer=True)

    # change the epochs for amount of times it runs through, batch_size for amount of pictures processed for each load
    model.fit(x_train, y_train, epochs=20, batch_size=64)

    # Reshaped for matplotlib, change the x_test to have different numbers
    img = x_train[50128].reshape(28, 28)
    from keras.datasets import mnist

    plt.imshow(img)
    plt.show()

    # Reshaped for model prediction
    img = img.reshape(-1, 28, 28, 1)
    out = model.predict(img)

    print("Guess: " + str(np.argmax(out)))
# +=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+
#         End of Keras/Mnist Recognition, Start of PyGame Drawing Window
# +=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+
# Initializing PyGame
pygame.init()

# Setting the parameters for the PyGame Window
width = 640
height = 640
radius = 35

# Setting RGB Values for colors
black = [0, 0, 0]
white = [255, 255, 255]
red = [255, 0, 0]
green = [0, 255, 0]

# Setting up the display/screen of the game
gameDisplay = pygame.display.set_mode((width, height))

# Setting the caption for the PyGame Window
pygame.display.set_caption('Number Drawing Window')

# Function for predicting the digit
def digit_predict(gameDisplay):
    # Here we put image to MNIST:
    data = pygame.image.tostring(gameDisplay, 'RGBA')
    img = Image.frombytes('RGBA', (width, height), data)
    img = img.resize((28, 28))
    imgobj = np.asarray(img)
    imgobj = cv2.cvtColor(imgobj, cv2.COLOR_RGB2GRAY)
    (_, imgobj) = cv2.threshold(imgobj, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    imgobj = imgobj/255
    b = model.predict(np.reshape(imgobj,[1, imgobj.shape[0], imgobj.shape[1],1]))
    ans = np.argmax(b)
    print("Predicted Value: ",ans)
    return ans

# Setting the font and colors for the return text
def textObjects(text, font):
	textSurface = font.render(text, True, red)
	return textSurface, textSurface.get_rect()

# Displaying the number guess
def message_display(text, locx, locy,size):
	largeText = pygame.font.Font('freesansbold.ttf', size) # Font(font name, font size)
	TextSurf, TextRec = textObjects(text, largeText)
	TextRec.center = (locx,locy)
	gameDisplay.blit(TextSurf, TextRec)
	pygame.display.update()

# Making a game loop that sets up the window
def game_loop():
    game_exit = False
    # Makes screen black
    gameDisplay.fill(black)
    pygame.display.flip()
    tick = 0
    tock = 0
    start_draw = False
    while not game_exit:
        if tock - tick >= 2 and start_draw:
            predVal = digit_predict(gameDisplay)
            gameDisplay.fill(black)
            message_display("Guess: "+str(predVal), int(width/2), int(height/2), 20)
            time.sleep(2)
            gameDisplay.fill(black)
            pygame.display.flip()
            tick = 0
            tock = 0
            start_draw = False
            continue

        tock = time.perf_counter()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_exit = True
        # Sets drawing mechanism
        if pygame.mouse.get_pressed()[0]:
            spot = pygame.mouse.get_pos()
            pygame.draw.circle(gameDisplay, white, spot, radius)
            pygame.display.flip()
            tick = time.perf_counter()
            start_draw = True

game_loop()
pygame.quit()