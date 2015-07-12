To do after provisioning:

make sure you can ssh to localhost without password:
ssh-keygen -t dsa -P '' -f ~/.ssh/id_dsa 
cat ~/.ssh/id_dsa.pub >> ~/.ssh/authorized_keys

start hadoop:
hadoop namenode -format
bash $HADOOP_HOME/bin/start-all.sh

setup sqoop:
sudo mkdir /usr/local/sqoop
sudo mv sqoop-1.4.6.bin__hadoop-1.0.0/ /usr/local/sqoop/
sudo mkdir /usr/lib/sqoop/
cd /usr/local/sqoop/sqoop-1.4.6.bin__hadoop-1.0.0/
sudo mv ./* /usr/lib/sqoop/
sudo emacs ~/.bashrc
     add these lines to the end:
     	 export SQOOP_HOME=/usr/lib/sqoop
	 export PATH=$PATH:$SQOOP_HOME/bin

get gateway ip to talk to mysql:
HOST_IP="$(netstat -rn | grep "^0.0.0.0 " | cut -d " " -f10 | sed -n '2p')"

sample sqoop command:
sqoop-import --connect jdbc:mysql://$HOST_IP/machine_learning --username <MYSQL USERNAME> --password <MYSQL PASSWORD> --table uni_lin_reg_data --target-dir /import/uni_lin_reg_data --columns "population, profit"

get scipy/numpy/pandas/etc
sudo apt-get install python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose

set Python path:
PYTHONPATH="${PYTHONPATH}:/usr/lib/python2.7/dist-packages/"

set up Spark:
wget http://d3kbcqa49mib13.cloudfront.net/spark-1.4.0-bin-hadoop1.tgz

tar zxvf spark-1.4.0-bin-hadoop1.tgz

export SPARK_HOME=<SPARK DIRECTORY>

export PATH=$SPARK_HOME/bin:$PATH

