# Predicting Hurricanes

### 2/28/2021 Note: I have recently returned to this project to make use of NOAA's Big Data Project's offerings on both AWS and GCP, which weren't available when I started the series. Please check out the "2021 update" folder and included code / notebooks.

### Here's the code associated with my "Predicting Hurricanes" YouTube series. https://www.youtube.com/channel/UCPmLClJE0GmnZ4e7sW_Fu7A

Far too many tutorials on using Machine Learning out there make it seem like your whole job is just taking a nice pre-processed dataset that you were simply given and running it through the latest algorithm of the week. And the job title "Data Sceientist" has been used to label pretty much anyone working with Machine Learning or even basic statistics these days. Many don't realize that building a system, testing its performance, making tweeks to it and retesting is in fact a closer description of the core engineering design process, rather than the scientific method, which stops at the conclusion of a single experiment. Applying research to design new systems that meet a specific set of requirements is engineering, not science. However, people with the title "Machine Learning Engineer" are rarely the ones designing the neural network or even the feature engineering. Companies continue to label the architects of prototype solutions "Data Scientists," which I sort of see as a misnomer, leading to confusion in terms of how to go about solving these sorts of problems, and ultimately part of the reason why I think so many "AI" initiatives fail. People with expertise in tensorflow are often hired on as either "Data Scientists" and given vague goals with fuzzy timelines, or brought on as "Machine Learning Engineers" to take the random successful projects of those Data Scientists and scale them up. There are taught to follow very basic steps to apply ML to problems, such as "clean the data, test these different approaches, etc." There's far too little focus on how to think about the problem at a more abstract level and find new datasets, or perform rigorous feature engineering (such as feature crosses) to handle non-linearities better. If more companies viewed Machine Learning as a tool that could be used to achieve a set of specific objectives, and then hired engineers to come in and apply that tool to that objective while meeting a set of clearly defined requirements, there would be a lot more success. 

This series attempts to teach you how to do some of that, at least as much as something as short as a YouTube series ultimately can. I use the "core engineering design process" to tackle a problem using Machine Learning and raw, uncleaned data. This process is used effectively by all disciplines of engineering, from mechanical and chemical to electrical and computer. Following this is how we figure out what precise problem needs to be solved, define our design requirements, research existing solutions to figure out what's been done and how we might be able to improve it or apply it in a different way, and then design a solution that will actually solve the problem. Randomly grabbing a dataset and trying a million things with it might get you a usable solution as well (and on the surface sounds like it fits the old "fail fast and fail often" philosophy of Silicon Valley startups), but following the engineering design process will ensure that you maintain a bigger picture strategy that will likely bring you to a more creative and interesting solution that works far better in the end. And ultimately, this is what many of those tech startups are actually doing, at least the successful ones. There's a lot of work that has to get done to set up the design/build phase, but once that whappens, we build prototype after prototype... iterating and honing in on the a final design until our performance metrics reach the desired levels. This process more than anything else is the reason why our technology has seen such a drastic increase in the rate of improvement over the past hundred years.

So that's what this series is all about... applying the engineering design process to solve big problems with the coolest new tools. And we're starting off with hurricanes. If you're interested in getting your hands dirty with me as we apply this process to the problem of forecasting hurricanes using machine learning, strap in and subscribe. 

Just a few notes up front: 
1. Instead of acting like a standard repository with the latest version being reflected by what you see here, I'm going to keep the code I use from each episode in its associated folder. So, "Episode_5" may have a newer version of the same file contained in "Episode_4" but that's just so you can follow along with the series. I'll include a readme in each episode folder to help aid you in using each file, but for the best experience, you should watch the associated episode.
1. Coding doesn't really start until Episode 4, which is why that's the first folder. I wanted to emphasize the work that goes into setting up a big problem like this, and so episodes 1-3 discuss the early steps of the core engineering design process.
1. Also, the datasets I use here are massive, and so I will not be posting any of that here. If you want to follow along, you'll have to download it yourself using the ftp sites and Python scripts I've written. Just a heads up though, I've basically filled up an entire 4TB hard drive, and I've only scratched the surface of the European data.
1. I will be using Python with Keras/Tensorflow. I'm not going to be teaching Python at all, and I'm not going to be focusing too much on how many of these algorithms work. There are a number of awesome free courses out there that you can take to teach you all that. I'd recommend Andrew Ng's Machine Learning Course on Coursera and David Silver's course on YouTube as a starting point. Instead, we're going to focus on the stuff that isn't really covered in any of the other courses I've seen out there. Namely, setting up the problem, working with real-world data, and training with utterly massive datasets.

If you're still onboard, you'll need a few things.

## MySQL

Available here: https://www.mysql.com/downloads/

Make sure that you also install MySQL Workbench. Mine is installed in a Windows environment, but I think it should be pretty much the same if you install in Linux. I usually install the mysqlclient and mysql-connector-c python libraries via Conda as well. There's a lot of documentation available to help you get started, but it's very straightforward if you've ever used any type of SQL before.

## Anaconda (aka "Conda")

This is what you'll generally want to use to install Python and manage whatever environments and associated installed packages you want to set up. It's just so much easier to do this with Conda than any other distribution I've found. You won't be able to find every package in Conda, but the core ones, like numpy, Pandas, sqlalchemy, etc. are all there, and Conda makes sure that all the versions are compatible. There are almost always instances where you'll have to install additional packages. Just do those at the end, after everything that's in Conda. We'll be using PyGrib, for instance, which you'll need to install with pip. Installing these last doesn't guarantee you won't "break" anything in your environment, but it minimizes the risk. With that said, do make sure you set up a second environment for all of this. Never use just the base environment. It's very difficult to reset it if something goes wrong, whereas it's very easy to delete a broken environment and start over. 

One other thing I like about Conda is it comes with Spyder, which is a pretty decent IDE that's great for working with data. A problem with using just a text editor with a terminal / command prompt is that it's annoying to view large datasets or even samples. With Spyder, you can view all your active variables in the upper right corner, and then simply double click one, like a numpy array or Pandas DataFrame, and you'll be able to see the whole dataset, with numerical columns color coded by value to help you spot outliers. Also, you can easily create several iPython consoles to run numerous parallel instances of simple modules that you may have running. I use this when I set up clusters. You can alternatively use threading or multiprocessing to accomplish the same thing, but the problem there is printing output. There are ways to do that, but it's far easier to spot / debug issues that may only show up several hours into a run when each thread is running in its own iPython console, and if you code things up correctly, you won't necessarily have to start the whole experiment over again to fix the issue. Just fix it in each of the modules, and get them running again. As long as the central "learner" or whatever you want to call it hasn't choked on a bad input, you can keep running. I find this to be far safer when setting up a training run that could go for a week, and when you're working on something real-world and not just training a neural network to play a simple low resolution Atari game or learn on a cultivated, pre-processed sample dataset, you'll find this is really helpful.

## RabbitMQ and Pika

RabbitMQ is a simple queueing / messaging service, and Pika is the python package we use to access the queues. It's accomplishing much the same thing that you would get from a Spark implementation on a bigger cluster, but for this problem and with just two desktop computers to work with, I decided to go with this approach, at least for now. Running a queuing service like this lets numerous threads working on multiple computers on a LAN all send data in the form of numpy arrays or even Pandas DataFrames to a central queue. Pulling data out of that queue is very fast, especially compared with running a query on a SQL database. What this lets us do is to store our data in a central database and have multiple threads running queries on that database in parallel, pulling samples and putting them in the queue so that our central "learner" (i.e. our thread that's actually training a neural network or whatever machine learning algorithm we're working with) can pull out a batch without having to wait for the SQL database to do any matching and return results. This lets us more fully utilize both our CPU and GPU while training... enabling our CPU to continue preparing and loading data while our GPU works on training.

The only real problem I have with it is the documentation isn't really written in plain english, and there's almost too much of it so it's easy to get overwhelmed trying to get it set up. This is because it's a very powerful piece of software, that's highly configurable so you can set it up on multiple network configurations, but for those just trying to do it on a home LAN, it's definitely information overload. All you really need to do is ensure that every computer that's using this has the same cookie installed. And you might need to make sure that that cookie is saved in two locations on some computers, since the command line interface tool looks in a different folder location for some reason. The Clustering Guide tells you where to install these. https://www.rabbitmq.com/clustering.html. If you're only using two computers, just use the rabbitmq command prompt (in windows) to join one to the other:

> rabbitmqctl stop_app
> rabbitmqctl join_cluster <other computer name - like rabbit@computer_name>
> rabbitmqctl start_app

In Linux, it's the same thing, just with 'sudo' (in Ubuntu) before each line. You only need to do this on one machine. To make sure it's working, you can use 'cluster_status' but I highly recommend instead installing the web interface tool by typing:

> rabbitmq-plugins enable rabbitmq_management

Then, you just go to http://localhost:15672 in your web browser. You can probably log in the first time with guest/guest, but check out this thread for more information. https://stackoverflow.com/questions/22850546/cant-access-rabbitmq-web-management-interface-after-fresh-install/22854222#22854222. I generally setup my own account / password with the admin tag.

## Considerations Moving Forward

There are many other ways of setting up a cluster to train a neural network, and many more to leverage multi-threading one a single machine to get some of the same benefits. If you only have one computer, Keras offers a nice solution called "fit_generator" that will do pretty much the same thing (i.e. preparing the next sample with your extra CPU capacity while training with the GPU). However, be aware it only really works in Linux (per this issue: https://github.com/keras-team/keras/issues/10842), so if you intend to go that route, get yourself a nice Linux distribution. Ubuntu is usually the best place for newbies to start because it's widely distributed and you can find answers to just about any issue on StackOverflow. I'm personally starting to eye the new Intel Clear distibution though, which has been making the news lately by putting out benchmark results that seem almost too good to be true. You may want to check that out.

And then there's Spark / PySpark. This seems to be the most commonly used platform amongst the bigger companies working with Machine Learning these days. It's pretty compelling.. allowing you to store data across a cluster instead of simply using a cluster to increase processing capability the way I do here. With Spark, you can cache a huge amount of your data in memory across the cluster and work with it much faster using dataframes not too dissimilar from what we use in Pandas. And it can even be used to accelerate MySQL queries by utilizing more threads to pull that data, or pre-fetch the most commonly used tables to memory. I'll certainly be looking into this more in the coming weeks... later episodes may make use of it if I can find an easy way to integrate it with this project. Otherwise, I'll probably feature it in a future season.




