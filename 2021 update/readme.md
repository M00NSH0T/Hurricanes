# Check Out the Jupyter Notebook

This is a work in progress. However, it's already quite long, so you may need to click the link to reload it a couple time to get it to display in GitHub. Once I finish it, I'll probably cut it into a few pieces to address this. 

Currently, my notebook walks through how to tie into the AWS GEOS16 satellite data, along with historic hurricane track data, to create a set of nearly 9000 views of several hundred storms from space. I combine satellie imagery using wavelengths outside the visible spectrum to ensure we see detail at night as well as during the day, and superimpose a slightly transparent view of these storms over a terrain map of the world, so our images also show coastline and other terrain detail usually obscured by cloud cover, and all but invisible at night. Here's a sample:

![image](https://github.com/M00NSH0T/Hurricanes/blob/master/2021%20update/storm_centered/centered_2017152N14262_20171523.png)

In the coming days, I will be adding a baseline tensorflow model that uses tf.data and Keras to create feature crosses of the track data and combine this with the image data to forecast future tracks. 

