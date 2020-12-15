import pygame
from math import sin, cos, radians
from random import randint
import numpy as np
import cv2
from scipy.ndimage import rotate
from math import sqrt
from collections import deque
import psutil
import os

p = psutil.Process(os.getpid())

try:

    p.nice(0)  # set>>> p.nice()10

except:

    p.nice(psutil.HIGH_PRIORITY_CLASS)

pygame.init()

# Sets the name and icon
pygame.display.set_caption('Self driving car using RL & Neural Networks 2.0')
game_icon = pygame.image.load('./assets/car.png')
pygame.display.set_icon(game_icon)

# Controls the resolution, 2 is optimal
screen_resolution_division = 2
screen = pygame.display.set_mode((1200//screen_resolution_division, 950//screen_resolution_division))
H = screen.get_height()
W = screen.get_width()

# Mirrored duplicate
screenlr = screen
screenrl = pygame.transform.flip(screen, True, False)

# Clock tick of the game
clock = pygame.time.Clock()

# Loads the images and resize
car  = pygame.image.load('./assets/car.png').convert_alpha() # Car image for display
car = pygame.transform.scale(car, (48//screen_resolution_division, 48//screen_resolution_division))
steering_wheel = pygame.image.load('./assets/steering_wheel.png').convert_alpha()
steering_wheel = pygame.transform.scale(steering_wheel, (120//screen_resolution_division, 120//screen_resolution_division))
speedometer = pygame.image.load('./assets/speedometer.png').convert_alpha()
speedometer = pygame.transform.scale(speedometer, (120//screen_resolution_division, 120//screen_resolution_division))
speed_indicator = pygame.image.load('./assets/speed_indicator.png').convert_alpha()
speed_indicator = pygame.transform.scale(speed_indicator, (90//screen_resolution_division, 90//screen_resolution_division))
course_background = pygame.image.load('./assets/course_background.png').convert()
course_background = pygame.transform.scale(course_background, (1200//screen_resolution_division, 950//screen_resolution_division))
mirrored_background = pygame.transform.flip(course_background, True, False)
finish_line = pygame.image.load('./assets/finish_line.png').convert()
finish_line = pygame.transform.scale(finish_line, (138//screen_resolution_division, 45//screen_resolution_division))

# Colors definition
RED = (255, 0, 0)
ORANGE = (255, 128, 0)
YELLOW = (255, 255, 73)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
COLOR_CHART = [RED, ORANGE, YELLOW, BLUE, GREEN]
BLACK = (0, 0, 0)
TRACK_COLOR = (162, 170, 159, 255)


# Fine adjustment for position of the finish line
x_finish_line_adjust = 36
y_finish_line_adjust = -40
FINISH_LINE = [(80 + x_finish_line_adjust) // screen_resolution_division,
               (436 + y_finish_line_adjust)// screen_resolution_division,
               W - finish_line.get_width() - (82 + x_finish_line_adjust) // screen_resolution_division,
               (436 + y_finish_line_adjust)// screen_resolution_division]

# Fine adjustment for car initial position
x_car_starting_pos_adjust = -5
y_car_starting_pos_adjust = -20

# Zoom window
zoom_window_size = 420
zoom_window_size = (zoom_window_size // screen_resolution_division, zoom_window_size //  screen_resolution_division)

frame_around_display = 35 # To prevent black areas when rotating

display_window_size = 160
state_size_window = (int(display_window_size // screen_resolution_division), int(display_window_size // screen_resolution_division)) # State used by DQN
display_window_size = (128//screen_resolution_division, 128//screen_resolution_division) # Displayed image

# Zoom window position
zoom_window_pos_x = W / 2 - display_window_size[0] / 2
zoom_window_pos_y = H - display_window_size[1] - 15

# Static images positioning
steering_wheel_pos_x = zoom_window_pos_x + 10 + display_window_size[0] * 1.5
steering_wheel_pos_y = H - steering_wheel.get_size()[1] - 15
speedometer_pos_x = zoom_window_pos_x - 10- display_window_size[0] * 1.5
speedometer_pos_y = H - speedometer.get_size()[1] - 15
speed_indicator_pos_x = speedometer_pos_x + speed_indicator.get_size()[0] / 2 - 24 // screen_resolution_division
speed_indicator_pos_y = speedometer_pos_y + speed_indicator.get_size()[1] / 2 - 24 // screen_resolution_division

pygame.font.init()

myfont = pygame.font.SysFont('Arial',20//screen_resolution_division)

# Global functions
# Image rotation by center of mass
def rot_center(image, angle):
    orig_rect = image.get_rect()
    rot_image = pygame.transform.rotate(image, angle)
    rot_rect = orig_rect.copy()
    rot_rect.center = rot_image.get_rect().center
    rot_image = rot_image.subsurface(rot_rect).copy()
    return rot_image


# Main class
class Car:
    def __init__(self, randomize_starting_point=False, randomize_mirror=False, trailing=False, show_stats=False, show_proximity=False):

        # Restart control
        self.restart = False

        # Starting configurations
        self.randomize_starting_point = randomize_starting_point
        self.randomize_mirror = randomize_mirror
        self.trailing = trailing
        self.show_stats = show_stats
        self.show_proximity = show_proximity

        # Randomize starting position
        if randomize_starting_point:

            r = randint(-50, 50)

        else:
            r = 0

        # Randonly mirror the track
        if randomize_mirror:
            self.mirror = randint(0,1)
        else:
            self.mirror = 0

        # Car drawing position (0,0)
        self.car_drawing_position_x = (168 + r + x_car_starting_pos_adjust) //screen_resolution_division
        self.car_drawing_position_y = (525 + y_car_starting_pos_adjust) //screen_resolution_division

        if self.mirror:
            self.car_drawing_position_x = W -  car.get_width() - (168 + r + x_car_starting_pos_adjust) //screen_resolution_division

        # Car position (center)
        self.car_pos_x = int(self.car_drawing_position_x) + 24//screen_resolution_division
        self.car_pos_y = int(self.car_drawing_position_y) + 24//screen_resolution_division

        # Car configurations
        self.car_heading_angle = 0
        self.speed = 0
        self.top_speed = 5
        self.speed_step = 0.1
        self.steer_angle = 0
        self.max_steer_angle = 5
        self.steer_angle_step = 0.5

        # Agent actions and state
        self.action_labels = ["NOACT", "BRAKE_LEFT", "LEFT", "ACCEL_LEFT", "ACCEL", "ACCEL_RIGHT", "RIGHT", "BRAKE_RIGHT", "BRAKE"]
        self.actions = [range(len(self.action_labels))]
        self.action_size = len(self.action_labels)

        # Agent reward sensors
        self.sensors_length = 160//screen_resolution_division
        self.sensors_angles = range(-40, 41, 5)
        self.sensors = [[0,0,0,0]] * len(self.sensors_angles) # x, y, x', y'
        self.state_size = np.array([0] * (len(self.sensors_angles) + 2)).shape[0]
        # self.state_size = np.zeros((state_size_window[0], state_size_window[1], 1)).shape

        # Action taken colors to show on trail
        self.act_labels_color = [(255,255,255), # NOACT
                               YELLOW, #BRAKE LEFT
                               BLUE, # LEFT
                               ORANGE, # ACCEL LEFT
                               GREEN, # ACCEL
                               ORANGE, # ACCEL RIGHT
                               BLUE, # RIGHT
                               YELLOW, # BRAKE RIGHT
                               RED] # BRAKE

        # Action, position and colors list to show on screen
        self.trail = deque(maxlen=5000)
        
        # Statistics
        self.run_steps = 0
        self.avg_speed = 0
        self.reward = 0

        # Run only in the startup
        try:
            if self.total_steps == 0:
                pass
        except:
            self.total_steps = 0
            self.total_reward = 0
            self.resets_number = 0
            self.total_avg_speed = 0


    def calculate_reward(self):

        # Mirrors background if set
        if self.mirror:
            background = mirrored_background
        else:
            background = course_background

        # Gets pixel color where car center is at and converts to a scalar to compare
        pixel_color_pos_car  = background.get_at((self.car_pos_x,self.car_pos_y))
        pixel_color_to_scalar_value = sum([abs(TRACK_COLOR[z] - pixel_color_pos_car[z]) for z in range(len(TRACK_COLOR))])

        # If out of the track
        if pixel_color_to_scalar_value > 50:
            self.restart = True # Reset position
            self.resets_number += 1
            return -50

        # Reward calculation according to sensors
        sens = (round(sum([(1-s) for s in self.sensors_detection_distance]), 2)) / len(self.sensors_angles) * 20
        speed = (self.speed - self.top_speed / self.top_speed) / 2

        # If stopped, bad behavior, if moving behavior depends on the sensors
        if self.speed == 0:
            self.reward = -25
        else:
            self.reward = -sens + speed

        # Rounds to prevent unwanted behaviour
        self.reward = round(self.reward, 4)
        
        # Accumulates total reward
        self.total_reward += self.reward

        return self.reward


    def sensors_calc(self):

        # Mirrors background if set
        if self.mirror:
            background = mirrored_background
        else:
            background = course_background

        # Calculates the sensors positioning taking into account sin and cos
        for i in range(len(self.sensors_angles)):
            s = sin(radians(self.car_heading_angle + self.sensors_angles[i]))
            c = cos(radians(self.car_heading_angle + self.sensors_angles[i]))
            self.sensors[i] = [self.car_pos_x, self.car_pos_y, int(self.car_pos_x + self.sensors_length * s), int(self.car_pos_y + self.sensors_length * c)]

        self.sensors_detection_distance = [self.sensors_length] * len(self.sensors_angles)

        # Calculates the point where the sensor enters in contact with out of the track
        for j in range(len(self.sensors_angles)):
            self.vec_x = np.linspace(self.sensors[j][0], self.sensors[j][2], self.sensors_length)
            self.vec_y = np.linspace(self.sensors[j][1], self.sensors[j][3], self.sensors_length)

            if (self.vec_x[0] - self.vec_x[-1]) < 0: self.vec_x.sort()
            if (self.vec_y[0] - self.vec_y[-1]) < 0: self.vec_y.sort()

            for i in range(self.sensors_length):
                pixel_color_sensor_pos  = background.get_at((int(self.vec_x[i]), int(self.vec_y[i])))
                pixel_colos_to_scalar_value = sum([abs(TRACK_COLOR[x] - pixel_color_sensor_pos[x]) for x in range(len(TRACK_COLOR))])

                if pixel_colos_to_scalar_value > 50:
                    self.sensors_detection_distance[j] = sqrt((self.vec_x[i] - self.car_pos_x)**2 + (self.vec_y[i] - self.car_pos_y)**2)
                    break

        # Calculates module of the distance
        for i in range(len(self.sensors_angles)):
            temp_length = sqrt((self.sensors[i][3] - self.sensors[i][1])**2 +
                               (self.sensors[i][2] - self.sensors[i][0])**2)
            self.sensors_detection_distance[i] = min(1, round(((self.sensors_detection_distance[i])/ (temp_length-5)), 4))


    def draw_stats(self):

        # If set, shows the reward on bottom left corner
        if self.show_stats:
            textsurface = myfont.render('TotalReward:{}   Total Steps:{}   Total avg speed:{}   Reward:{}   Steps:{}  Avg speed: {}   Resets:{}'
            .format(round(self.total_reward, 2), self.total_steps,round(self.total_avg_speed, 2), self.reward, self.run_steps, round(self.avg_speed, 2), self.resets_number), False, BLACK)
            screen.blit(textsurface,(8 // screen_resolution_division, H-24//screen_resolution_division))


    def draw_proximity_display(self):

        # Displays rgb circles according to proximity sensors
        if self.show_proximity:
            colors =  [max(0, int(d * 10) //2 -1) for d in self.sensors_detection_distance]
            colors.reverse()
            for i in range(len(self.sensors_angles)):
                l = len(self.sensors_angles)
                color = COLOR_CHART[colors[i]]
                pygame.draw.circle(screen, color, (W // 2 - l * 10 // screen_resolution_division + i * 20 // screen_resolution_division, H - 180 // screen_resolution_division), 6 // screen_resolution_division)


    def draw_sensors(self):
        # Draws sensors, position from x,y to x',y'
         for i in range(len(self.sensors_angles)):
                s = sin(radians(self.car_heading_angle + self.sensors_angles[i]))
                c = cos(radians(self.car_heading_angle + self.sensors_angles[i]))
                pygame.draw.line(screen,YELLOW,(self.sensors[i][0] + 15 * s, self.sensors[i][1] + 15 * c),
                                                (self.sensors[i][2], self.sensors[i][3]), 1)


    def draw_trail(self):
        # Draws action trail if set
        if self.trailing:
            myfont = pygame.font.SysFont('Arial', 50)
            for x,y,c in self.trail:
                textsurface = myfont.render('.', False,c)
                screen.blit(textsurface,(x-6,y-42))


    def store_trail(self, action):
        self.trail.append([self.car_pos_x, self.car_pos_y, self.act_labels_color[action]])


    def steer(self, env_action):
        if "LEFT" in env_action:
            if self.steer_angle < self.max_steer_angle:
                self.steer_angle += self.steer_angle_step

        elif "RIGHT" in env_action:
            if self.steer_angle > -self.max_steer_angle:
                self.steer_angle -= self.steer_angle_step

        else:
            if self.steer_angle > 0:
                self.steer_angle -= self.steer_angle_step
            elif self.steer_angle < 0:
                self.steer_angle += self.steer_angle_step

        if self.speed != 0:
            self.car_heading_angle += self.steer_angle

        # Rounds to prevent unwanted behaviour
        self.steer_angle = round(self.steer_angle, 2)



    def accel_brake(self, env_action):
        if "ACCEL" in env_action:
            if self.speed < self.top_speed:
                    self.speed += self.speed_step * 2

        elif "BRAKE" in env_action:
            if self.speed > 0:
                self.speed -= self.speed_step * 2

        else:
            if self.speed > 0:
                self.speed -= self.speed_step

        # Rounds and assures min value = 0 to prevent unwanted behaviour
        self.speed = max(0, round(self.speed, 2))

        # Current avg speed
        self.avg_speed = (self.avg_speed + self.speed) / 2
        
        # Total avg speed
        self.total_avg_speed = (self.total_avg_speed * self.total_steps + self.speed) / (self.total_steps + 1)


    def draw(self):

        # Mirrors background if set
        if self.mirror:
            background = mirrored_background
        else:
            background = course_background

        # Draw background
        screen.blit(background, (0,0))

        # Gets zoomed car image
        zoom = self.zoom()

        # Calculates proximity to out of the track
        self.sensors_calc()

        # Mirrors finish line position
        if self.mirror:
            i,j = FINISH_LINE[2], FINISH_LINE[3]
        else:
            i,j = FINISH_LINE[0], FINISH_LINE[1]

        # Draws finish line
        screen.blit(finish_line, (i, j))

        # Rotates car and draw
        car_ = rot_center(car, self.car_heading_angle)
        screen.blit(car_, (self.car_drawing_position_x, self.car_drawing_position_y))

        self.draw_sensors()
        self.draw_proximity_display()
        self.draw_stats()

        # Draw trail of actions on screen
        self.draw_trail()

        # Rotate with the car orientation always up
        zoom = rotate(zoom, -self.car_heading_angle, reshape=False)
        zoom = np.rot90(zoom, 3)
        zoom = np.flipud(zoom)

        # Threshold to correct image imperfections
        retval, zoom = cv2.threshold(zoom, 150, 255, cv2.THRESH_BINARY)

        # Changes black pixels to gray
        zoom[zoom == 0] = 127

        # Gets actual image to display, removing a small border from around
        zoom = zoom[frame_around_display : zoom.shape[0] - frame_around_display,
                    frame_around_display : zoom.shape[1] - frame_around_display]

        # Resizes the zoomed image for the DQN
        state = cv2.resize(zoom, (state_size_window))

        # Draws a small croshair on the center of the image
        state = cv2.line(state, (state.shape[0]//2-5, state.shape[1]//2), (state.shape[0]//2+5, state.shape[1]//2), (0, 0, 0), thickness=2)
        state = cv2.line(state, (state.shape[0]//2, state.shape[1]//2-5), (state.shape[0]//2, state.shape[1]//2+5), (0, 0, 0), thickness=2)

        # Creates, resizes and draws mini display for displaying on the bottom of the screen
        mini_display = cv2.resize(state, (display_window_size))
        mini_display_surface = pygame.Surface(display_window_size)
        pygame.surfarray.blit_array(mini_display_surface, mini_display)
        screen.blit(mini_display_surface, (zoom_window_pos_x, zoom_window_pos_y))

        # Rotate steering wheel and draw
        steering_wheel_ = rot_center(steering_wheel, self.steer_angle * 20)
        screen.blit(steering_wheel_, (steering_wheel_pos_x,steering_wheel_pos_y))

        # Draw speedometer
        screen.blit(speedometer, (speedometer_pos_x,speedometer_pos_y))

        # Rotates speedometer pointer and draw
        speed_indicator_ = rot_center(speed_indicator, 135 - self.speed * 50)
        screen.blit(speed_indicator_, (speed_indicator_pos_x,speed_indicator_pos_y))

        # Updates the screen image
        pygame.display.flip()

        # Corrects the image orientation pygame != np
        state = np.rot90(state, 3)
        state = np.fliplr(state)

        # Picks one layer, and normalize it by rescaling by 255 to feed the network
        state = state[:,:,0]
        state =  state / 255.

        return state.reshape(state.shape[0], state.shape[1], 1) # The neural network needs it to be this shape


    def move_car(self):
        # Car image positioning
        self.car_drawing_position_x += sin(radians(self.car_heading_angle)) * self.speed
        self.car_drawing_position_y += cos(radians(self.car_heading_angle)) * self.speed
        # Car image center positioning
        self.car_pos_x = int(self.car_drawing_position_x) + 24//screen_resolution_division
        self.car_pos_y = int(self.car_drawing_position_y) + 24//screen_resolution_division


    def create_zoom_frame(self, temp_array):
        upper_frame = 0
        lower_frame = 0
        right_frame = 0
        left_frame = 0

        # If the zoom image area falls outside the screen, stores by how much it exceeded
        if int(self.car_pos_y - zoom_window_size[0] / 2) < 0:
            upper_frame = int(self.car_pos_y - zoom_window_size[0] / 2)

        if int(self.car_pos_y + zoom_window_size[0] / 2) > H:
            lower_frame = int(self.car_pos_y + zoom_window_size[0] / 2) - H

        if int(self.car_pos_x + zoom_window_size[1] / 2) > W:
            right_frame = int(self.car_pos_x + zoom_window_size[1] / 2) - W

        if int(self.car_pos_x - zoom_window_size[1] / 2) < 0:
            left_frame = int(self.car_pos_x - zoom_window_size[1] / 2)

        # Crop the area around the car
        temp_array = temp_array[int(-(upper_frame) + self.car_pos_y - zoom_window_size[1] / 2):
                                int(-(lower_frame) + self.car_pos_y + zoom_window_size[1] / 2),
                                int(-(left_frame) + self.car_pos_x - zoom_window_size[0] / 2):
                                int(-(right_frame) + self.car_pos_x + zoom_window_size[0] / 2)]

        # Creates a frame with the size of how much it exceed in all directions
        output_image = np.ones((zoom_window_size[0], zoom_window_size[1], 3)) * 127

        # Overlay the cropped image on top of the frame
        output_image[abs(upper_frame) : zoom_window_size[1] - abs(lower_frame),
              abs(left_frame):zoom_window_size[0] - abs(right_frame), :] = temp_array

        return output_image.astype('uint8')


    def zoom(self):
        # Converts to image array
        string_image = pygame.image.tostring(screen, 'RGB')
        temp_surf = pygame.image.fromstring(string_image, (W, H),'RGB' )
        temp_array = pygame.surfarray.array3d(temp_surf)

        # Corrects the image orientation pygame != np
        temp_array = np.rot90(temp_array, 1)
        temp_array = np.flipud(temp_array)

        # Splits rgb, filter colors different from track color and rejoin
        b,g,r = cv2.split(temp_array)
        r[r != TRACK_COLOR[2]] = 127
        g[g != TRACK_COLOR[1]] = 127
        b[b != TRACK_COLOR[0]] = 127
        temp_array = cv2.merge((r,g,b))

        # Creates a frame around the image to end up the right size
        temp_array = self.create_zoom_frame(temp_array)

        return temp_array


    def run(self, action="NOACT"):
        pygame.event.get()

        # Accepts input as string or number
        if type(action) == str:
            env_action = action
            action = self.action_labels.index(action)
        else:
            env_action = self.action_labels[action]

        # Accumulates run steps
        self.run_steps += 1
        self.total_steps += 1

        # Main functions
        self.store_trail(action)
        self.steer(env_action)
        self.accel_brake(env_action)
        self.move_car()
        state = self.draw()

        # Check state and give reward
        reward = self.calculate_reward()

        # If flag set to restart, restarts the game
        if self.restart:
            self.__init__(self.randomize_starting_point, self.randomize_mirror, self.trailing, self.show_stats, self.show_proximity)

        # Concatenates sensors, speed and angle
        sensors_speed_angle = np.array(self.sensors_detection_distance + [round(self.speed/self.top_speed, 2)] +
                                       [round((self.steer_angle + self.max_steer_angle)/(self.max_steer_angle*2), 2)])

        # Reshapes for the network input
        sensors_speed_angle = sensors_speed_angle.reshape(1, -1)

        return state, reward, env_action, sensors_speed_angle


if __name__ == "__main__":

    env = Car(show_proximity=True)

    while True:

        # Get key press to control
        pressed = pygame.key.get_pressed()

        env_action = 'NOACT' # Starts the variable

        # Use pressed keys to control the environment
        if pressed[pygame.K_UP]:
            env_action = "ACCEL"

        elif pressed[pygame.K_DOWN]:
            env_action = "BRAKE"

        if pressed[pygame.K_RIGHT]:
            if env_action != 'NOACT':
                env_action = env_action +"_RIGHT"
            else:
                env_action = 'RIGHT'

        elif pressed[pygame.K_LEFT]:
            if env_action != 'NOACT':
                env_action = env_action +"_LEFT"
            else:
                env_action = 'LEFT'

        # Quits the game
        if pressed[pygame.K_q]: pygame.quit()

        # Output of the environment
        _, reward, action, state = env.run(env_action)
        print('Reward: {}  Action: {} State: {}'.format(reward, action, state))

        # Sets the game fps
        clock.tick(60)
