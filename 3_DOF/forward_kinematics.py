import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.animation import FuncAnimation

class RobotArm3D:
    def __init__(self, segment_lengths):
        self.segment_lengths = segment_lengths
        self.angles = [0, 0, 0]  # Initial angles for each joint, 

    def forward_kinematics(self, angles):
        points = []
        position = np.array([0.0, 0.0, 0.0])
        orientation = np.eye(3)  # Identity matrix for initial orientation
        points.append(position.copy())

        for i, (theta, length) in enumerate(zip(angles, self.segment_lengths)):
            theta_rad = np.radians(theta)
            if i == 0:
                rotation = np.array([
                    [np.cos(theta_rad), -np.sin(theta_rad), 0],
                    [np.sin(theta_rad), np.cos(theta_rad), 0],
                    [0, 0, 1]
                ])
            elif i == 1:
                rotation = np.array([
                    [np.cos(theta_rad), 0, np.sin(theta_rad)],
                    [0, 1, 0],
                    [-np.sin(theta_rad), 0, np.cos(theta_rad)]
                ])
            elif i == 2:
                rotation = np.array([
                    [np.cos(theta_rad), -np.sin(theta_rad), 0],
                    [np.sin(theta_rad), np.cos(theta_rad), 0],
                    [0, 0, 1]
                ])

            orientation = orientation @ rotation
            position += orientation @ (np.array([length, 0.0, 0.0]) if i < 2 else np.array([0.0, length, 0.0]))
            points.append(position.copy())

        return np.array(points)

    def update_angles(self, angles):
        self.angles = [max(-270, min(270, a)) for a in angles]

    def plot_arm_interactive(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        def on_key(event):
            if event.key == '1':
                self.angles[0] = max(-175, self.angles[0] - 5)
            elif event.key == '2':
                self.angles[0] = min(175, self.angles[0] + 5)
            elif event.key == '3':
                self.angles[1] = max(-175, self.angles[1] - 5)
            elif event.key == '4':
                self.angles[1] = min(175, self.angles[1] + 5)
            elif event.key == '5':
                self.angles[2] = max(-265, self.angles[2] - 5)
            elif event.key == '6':
                self.angles[2] = min(85, self.angles[2] + 5)
            self.update_angles(self.angles)
            update_arm()

        def update_arm():
            ax.cla()
            ax.set_xlim([-sum(self.segment_lengths), sum(self.segment_lengths)])
            ax.set_ylim([-sum(self.segment_lengths), sum(self.segment_lengths)])
            ax.set_zlim([-sum(self.segment_lengths), sum(self.segment_lengths)])
            points = self.forward_kinematics(self.angles)
            ax.plot(points[:, 0], points[:, 1], points[:, 2], 'o-', markersize=8)
            ax.scatter(points[-1, 0], points[-1, 1], points[-1, 2], c='orange', s=100)  # Orange ball at the end
            ax.set_title(f"3D Robot Arm Interactive Simulation\nX: {self.angles[0]}° Y: {self.angles[1]}° Z: {self.angles[2]}°")
            plt.draw()

        fig.canvas.mpl_connect('key_press_event', on_key)
        ax.mouse_init()  # Enable mouse rotation
        update_arm()
        plt.show()

# Example usage:
segment_lengths = [5, 5, 5]
robot_arm = RobotArm3D(segment_lengths)
robot_arm.plot_arm_interactive()
