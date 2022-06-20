## Assignment Assignment #04 - Create 3D geometry

## About this Assignment

This project is about creating a mesh 3D Object from a kinect recording

### Built with

Python: 3.9 <br>
Opencv: 4.5.5 <br>
Open3D: 0.15 <br>

### Libaries needed

install matplotlib <br>
install joblib

Here is a quick installation guide for open3d:
http://www.open3d.org/docs/release/getting_started.html

### This tutorial is inspired from

http://www.open3d.org/docs/release/tutorial/reconstruction_system/index.html

### Getting started

For this tutorial a recorded .mkv file is needed. This .mkv file is generated with a Azure Kinect recording. Follow this
installation guide if you want to record your own file.
http://www.open3d.org/docs/latest/tutorial/Basic/azure_kinect.html

In the folder

```python
examples / python / ReconstructionSystem / sensors /
```

you can run the script ```azure_kinect_recorder.py``` with the command

```python
python
examples / python / ReconstructionSystem / sensors / azure_kinect_recorder.py - -output
record.mkv
```

If you stop the recording the file is saved in the same folder like above.

### Extract the information about depth and color from the .mkv file.

To extract the depth and color map from every frame of the file you can use the command from the sensor folder

```python
python
azure_kinect_mkv_reader.py - -input
record.mkv - -output
frames
```

This will generate a folder named frames where the depth and color map is stored for every frame.

* depth
* color

### Quick overview

To generate a 3D object from the image and depth folder this command should be run:

```python
cd examples/python/reconstruction_system/python run_system.py[config_file][--make][--register][--refine][--integrate]
```

### [--make]

```python
python
run_system.py "sensors/frames/config.json" - -make
```

This command creates two files:
*config.json (configuration file)
*intrinsic.json (intrinsic parameters are saved)

* fragments with
    * RGBD Odometry -> The function reads a pair of RGBD images and registers the source_rgbd_image to the
      target_rgbd_image
    * Multiway registration -> The function make_posegraph_for_fragment builds a pose graph for multiway registration of
      all RGBD images in this sequence
    * and RGBD integration -> Once the poses are estimates, RGBD integration is used to reconstruct a colored fragment
      from each RGBD sequence

one fragment include a .ply file and a pose graph in a .json file for each depth and color map.

### [--register]

Once the fragments of the scene are created, the next step is to align them in a global space.

```python
python
run_system.py "sensors/frames/config.json" - -register
```

### [--refine]

The main function runs local_refinement and optimize_posegraph_for_scene. The first function performs pairwise
registration on the pairs detected by Register fragments. The second function performs multiway registration.

```python
python
run_system.py "sensors/frames/config.json" - -refine
```

### [-- integrate]

The final step of the system is to integrate all RGBD images into a single TSDF volume and extract a mesh as the result.

```python
python run_system.py "sensors/frames/config.json" --integrate
```

The final result is stored in the scene folder called "integrated.ply"