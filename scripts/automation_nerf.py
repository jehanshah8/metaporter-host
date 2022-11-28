import subprocess


path_to_data_folder = "" #should have subfolder called images
path_to_ngp = ""
command = f"cd {path_to_data_folder}; ls -l;python {path_to_ngp}/scripts/colmap2nerf.py --colmap_matcher exhaustive --run_colmap --aabb_scale 16;"

ret = subprocess.run(command, capture_output=True, shell=True)

print(ret.stdout.decode())