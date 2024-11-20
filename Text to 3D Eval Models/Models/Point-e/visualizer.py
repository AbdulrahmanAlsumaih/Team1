import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

def save_to_ply(pc, filename="output.ply"):
    points = pc.coords
    colors = np.stack([pc.channels[x] for x in ['R', 'G', 'B']], axis=1)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(filename, pcd)

def create_rotation_animation(pc, output_file="rotation.mp4", fps=30, duration=5):  # Reduced duration to 5 seconds
    fig = plt.figure(figsize=(10, 10), facecolor='white')
    ax = fig.add_subplot(111, projection='3d', facecolor='white')
    
    points = pc.coords
    colors = np.stack([pc.channels[x] for x in ['R', 'G', 'B']], axis=1)
    
    def update(frame):
        ax.clear()
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors)
        ax.view_init(elev=20, azim=frame * 2)  # Doubled rotation speed
        ax.set_xlim([-0.75, 0.75])
        ax.set_ylim([-0.75, 0.75])
        ax.set_zlim([-0.75, 0.75])
        ax.grid(False)
        ax.set_axis_off()
        
    frames = fps * duration
    anim = FuncAnimation(fig, update, frames=frames, interval=1000/fps)
    writer = animation.FFMpegWriter(fps=fps)
    anim.save(output_file, writer=writer)
    plt.close()

save_to_ply(pc)  # Saves as output.ply
create_rotation_animation(pc)  # Saves as rotation.mp4