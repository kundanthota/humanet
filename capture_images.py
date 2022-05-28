import numpy as np
import trimesh
import argparse
import os

def args():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--resolution', type = int, required = True,\
                        help='for 300x300 image enter 300')

    parser.add_argument('--gender', type = str, required = True,\
                        help='male or female')

    parser.add_argument('--path', type = str, required = True,\
                        help='path to the .obj files') 
    
    arguments = parser.parse_args()
    return arguments

def render(resolution, file_path, save_path):
    '''
    Captures the scene of the 3D human rotated along the y-axis at 90 degrees and
    saves the files as .png pictures
    '''

    for subject in os.listdir(file_path):
        
        _file = os.path.join(file_path, subject)
        
        mesh = trimesh.load(_file)
        
        mesh.visual.vertex_colors =[205, 205, 205, -50]
        mesh.visual.face_colors =[205, 205, 205, -50]
        
        scene = mesh.scene()

        # a 90 degree homogeneous rotation matrix around
        # the Y axis at the scene centroid
        rotate = trimesh.transformations.rotation_matrix(
            angle=np.radians(90.0),
            direction=[0, 1, 0],
            point=scene.centroid)
        
        _id = subject.split(".")[0].split("_")[-1]

        try:
            os.mkdir(os.path.join(save_path, str(_id)))
        except:
            pass
        
        try:
            os.mkdir(os.path.join(save_path, _id, str(resolution)))
        except:
            pass
        
        _folder = os.path.join(save_path, _id, str(resolution))

        for side in ["front", "right"]:

            
            try:
                # file name to save
                file_name = os.path.join(_folder, f'{side}.png')

                # save a render of the object as a png
                png = scene.save_image(resolution=[resolution, resolution], visible=False)
                with open(file_name, 'wb') as f:
                    f.write(png)
                    f.close()

            except BaseException as E:
                pass
            
            # rotate the camera view transform
            camera_old, _geometry = scene.graph[scene.camera.name]
            camera_new = np.dot(rotate, camera_old)

            # apply the new transform
            scene.graph[scene.camera.name] = camera_new

def main():
    arguments = args()
    resolution = arguments.resolution
    gender = arguments.gender

    file_path = arguments.path
    
    try:
        os.mkdir("data")
    except:
        pass
    
    try:
        os.mkdir(f"data/{gender}")
    except:
        pass
    
    save_path = f"data/{gender}"
    render(resolution, file_path, save_path)
    

if __name__ == '__main__':
    main()
    