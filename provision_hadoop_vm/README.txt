To do after provisioning:

make sure you can ssh to localhost without password:
ssh-keygen -t dsa -P '' -f ~/.ssh/id_dsa 
cat ~/.ssh/id_dsa.pub >> ~/.ssh/authorized_keys

start hadoop:
hadoop namenode -format
bash $HADOOP_HOME/bin/start-all.sh

sample sqoop command:
sqoop-import --connect jdbc:mysql://10.0.2.2/machine_learning --username <MYSQL USERNAME> --password <MYSQL PASSWORD> --table uni_lin_reg_data --target-dir /import/uni_lin_reg_data --columns "population, profit"

set up Spark:
wget http://d3kbcqa49mib13.cloudfront.net/spark-1.4.0-bin-hadoop1.tgz

tar zxvf spark-1.4.0-bin-hadoop1.tgz

export SPARK_HOME=<SPARK DIRECTORY>

export PATH=$SPARK_HOME/bin:$PATH

