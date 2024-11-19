import numpy as np

def camera_to_world_matrix(azimuth, elevation, distance=1.0):
    # Convert angles from degrees to radians
    azimuth = np.radians(azimuth)
    elevation = np.radians(elevation)
    
    # Azimuth rotation matrix (around y-axis)
    R_y = np.array([
        [np.cos(azimuth), 0, np.sin(azimuth)],
        [0, 1, 0],
        [-np.sin(azimuth), 0, np.cos(azimuth)]
    ])
    
    # Elevation rotation matrix (around x-axis)
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(elevation), -np.sin(elevation)],
        [0, np.sin(elevation), np.cos(elevation)]
    ])
    
    # Combined rotation matrix
    R = np.dot(R_y, R_x)
    
    # Translation vector
    T = np.array([
        distance * np.cos(elevation) * np.sin(azimuth),
	distance * np.sin(elevation),
        distance * np.cos(elevation) * np.cos(azimuth)
    ])
    
    # Camera-to-world transformation matrix
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = T
    
    return R, T

if __name__ == '__main__':
    # Example usage
    R1, T1 = camera_to_world_matrix(30, 20)
    print("R1:")
    print(R1)
    print("T1:")
    print(T1)

    R2, T2 = camera_to_world_matrix(90, -10)
    print("R2:")
    print(R2)
    print("T2:")
    print(T2)

    R3, T3 = camera_to_world_matrix(150, 20)
    print("R3:")
    print(R3)
    print("T3:")
    print(T3)

    R4, T4 = camera_to_world_matrix(210, -10)
    print("R4:")
    print(R4)
    print("T4:")
    print(T4)

    R5, T5 = camera_to_world_matrix(270, 20)
    print("R5:")
    print(R5)
    print("T5:")
    print(T5)

    R6, T6 = camera_to_world_matrix(330, -10)
    print("R6:")
    print(R6)
    print("T6")
    print(T6)



