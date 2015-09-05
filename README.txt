To do after provisioning:

make sure you can ssh to localhost without password:
ssh-keygen -t dsa -P '' -f ~/.ssh/id_dsa 
cat ~/.ssh/id_dsa.pub >> ~/.ssh/authorized_keys

start hadoop:
hadoop namenode -format
bash $HADOOP_HOME/bin/start-all.sh

sample sqoop command:
sqoop-import --connect jdbc:mysql://10.0.2.2/machine_learning --username <MYSQL USERNAME> --password <MYSQL PASSWORD> --table uni_lin_reg_data --target-dir /import/uni_lin_reg_data --columns "Population, Profit"

do spark:
wget http://mirrors.advancedhosters.com/apache/spark/spark-1.4.0/spark-1.4.0.tgz
tar zxvf spark-1.4.0.tgz
cd spark-1.4.0
MAVEN_OPTS="-Xmx2g -XX:MaxPermSize=512M -XX:ReservedCodeCacheSize=512m" mvn -Dhadoop.version=1.2.1 -Phadoop-1 -DskipTests clean package
echo 'export SPARK_HOME=/home/vagrant/spark-1.4.0' >> /home/vagrant/.bashrc
echo 'export PATH=$SPARK_HOME/bin:$PATH' >> /home/vagrant/.bashrc

must create a file in top level directory of this repository called my_sql_conn.py declaring the mySQLconn variable.  Should look like the following:

import MySQLdb

mySQLconn = MySQLdb.connect(host=<db hostname, sometimes "localhost">,
                        	user=<db username>,
                        	passwd=<db password>,
                        	db=<db name>
                        	)
