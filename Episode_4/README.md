Here's the code for Episode 4. https://www.youtube.com/watch?v=qhQqmGt_Z7E&t=3609s

A few notes:
The GAN I used can be found here: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
To make it work with the NOAA satellite data (which you can download with "sat_image_downloader.py"), you need to just run "combine_pairs_modified.py." You will also need the list of dates that you get from the track data, so that needs to be imported into your SQL database first, or you need to modify the file to read that from the csv file here directly, and parse out the unique dates. Once you combine the images into the side by side set as shown in the video, you then just need to follow the instructions provided in the original repository. 

When you run train.py, I recommend these arguments:
>python train.py --dataroot ./datasets/satellite_imagery --name output --model pix2pix --direction AtoB --crop_size 1024 --load_size 1024 --no_flip

Again, be aware some of this requires you have a MySQL database setup. At a minimum, the satellite image downloader will use that to pull the unique date that correspond to active hurricanes from the track data, so we know what to grab. That data needs to be first populated using the import_track_data.py module. I included the track data as a csv here since it's only 6.6MB.

The CNN I ran through requires that you setup RabbitMQ, which is faily straight forward. However, if you want to run this on more than one machine (which is what I did to get it to complete relatively quickly), you need to setup a cluster. I found the easiest way to do this was via the RabbitMQ Command Prompt using the "rabbitmqctl join_cluster <cluster name>" command, but doing this requires that the erlang cookie be the same on both machines. You need to copy that from one machine to the other. It took me a solid day to figure that one out.
  
In the future, I may work on transitioning this to a PySpark approach instead, but for now, this is what I went with. It works alright, and the RabbitMQ web management tool is helpful for keeping an eye on performance. I generally try to get just enough sample generator instances going that the queue never quite fills up to capacity.

