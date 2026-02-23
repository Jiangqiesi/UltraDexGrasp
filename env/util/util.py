import numpy as np
from scipy.spatial.transform import Rotation as R


def save_rgb_images_to_video(images, output_filename, fps=30):
    import subprocess
    height, width, layers = images[0].shape
    command = ['ffmpeg',
               '-y',
               '-f', 'rawvideo',
               '-vcodec', 'rawvideo',
               '-s', f'{width}x{height}',
               '-pix_fmt', 'rgb24',
               '-r', str(fps),
               '-i', '-',
               '-c:v', 'libx264',
               '-pix_fmt', 'yuv420p',
               output_filename]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for image in images:
        process.stdin.write(image.tobytes())
    process.stdin.close()
    process.wait()

def calculate_angle_between_quat(q1, q2_array):
    q1_rot = R.from_quat(q1, scalar_first=True)
    q2_rot = R.from_quat(q2_array, scalar_first=True)
    angle = (q1_rot.inv() * q2_rot).magnitude()

    return angle

def calculate_fovy(fy, image_height):
    fovy_rad = 2 * np.arctan(image_height / (2 * fy))
    fovy_deg = np.degrees(fovy_rad)

    return fovy_deg

fy = 200
img_height = 480

fovy = calculate_fovy(fy, img_height)
print(fovy)