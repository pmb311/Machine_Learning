
# do maven
wget ftp://mirror.reverse.net/pub/apache/maven/maven-3/3.1.1/binaries/apache-maven-3.1.1-bin.tar.gz
tar -xvf apache-maven-3.1.1-bin.tar.gz
mv apache-maven-3.1.1 maven
# do environment settings
sudo sh -c "cat >> /etc/environment" <<'EOF'
export HADOOP_HOME=/home/vagrant/hadoop-1.2.1
export MAVEN_HOME=/home/vagrant/maven
export JAVA_HOME=/usr/lib/jvm/java-6-openjdk/jre
PATH=$PATH:$JAVA_HOME/bin:$MAVEN_HOME/bin:$HADOOP_HOME/bin
export PATH
EOF
# set env var's
source /etc/environment
# edit hadoop env & config
sed -i 's/# export JAVA_HOME=\/usr\/lib\/j2sdk1.5-sun/export JAVA_HOME=\/usr\/lib\/jvm\/java-6-openjdk\/jre/g' $HADOOP_HOME/conf/hadoop-env.sh
sed -i '/^<configuration>$/ s:$:\n<property>\n\t<name>hadoop.tmp.dir</name>\n\t<value>/home/vagrant/tmp</value>\n\t<description>A base for other temporary directories.</description>\n</property>:' $HADOOP_HOME/conf/core-site.xml
sed -i '/^<configuration>$/ s:$:\n<property>\n\t<name>hadoop.tmp.dir</name>\n\t<value>/home/vagrant/tmp</value>\n\t<description>A base for other temporary directories.</description>\n</property>\n<property>\n\t<name>fs.default.name</name>\n\t<value>hdfs\://localhost\:54310</value>\n\t<description>The name of the default file system.</description>\n</property>:' $HADOOP_HOME/conf/core-site.xml
sed -i '/^<configuration>$/ s:$:\n<property>\n\t<name>mapred.job.tracker</name>\n\t<value>localhost\:54311</value>\n\t<description>The host and port that the MapReduce job tracker runs.</description>\n</property>\n<property>\n\t<name>mapred.child.java.opts</name>\n\t<value>-Xmx256M</value>\n</property>:' $HADOOP_HOME/conf/mapred-site.xml
sed -i '/^<configuration>$/ s:$:\n<property>\n\t<name>dfs.replication</name>\n\t<value>1</value>\n\t<description>Default block replication.</description>\n</property>:' $HADOOP_HOME/conf/hdfs-site.xml

# do sqoop
wget ftp://mirrors.sonic.net/apache/sqoop/1.4.6/sqoop-1.4.6.bin__hadoop-1.0.0.tar.gz
tar -zxvf sqoop-1.4.6.bin__hadoop-1.0.0.tar.gz
sudo mkdir /usr/local/sqoop
sudo mv /home/vagrant/sqoop-1.4.6.bin__hadoop-1.0.0/ /usr/local/sqoop/
sudo mkdir /usr/lib/sqoop/
cd /usr/local/sqoop/sqoop-1.4.6.bin__hadoop-1.0.0/
sudo mv ./* /usr/lib/sqoop/
cd /home/vagrant/
# get mysql driver for sqoop
wget http://dev.mysql.com/get/Downloads/Connector-J/mysql-connector-java-5.1.36.tar.gz
tar zxvf mysql-connector-java-5.1.36.tar.gz
sudo mv mysql-connector-java-5.1.36/mysql-connector-java-5.1.36-bin.jar /usr/lib/sqoop/
#clean up
rm -r mysql-connector-java-5.1.36
sudo rm *.tar.gz
sudo chown vagrant:vagrant *
sudo mv /usr/lib/jvm/java-6-openjdk-amd64/ /usr/lib/jvm/java-6-openjdk
#edit .bashrc
echo 'source /etc/environment' >> /home/vagrant/.bashrc
echo 'export SQOOP_HOME=/usr/lib/sqoop' >> /home/vagrant/.bashrc
echo 'export PATH=$PATH:$SQOOP_HOME/bin' >> /home/vagrant/.bashrc
echo 'export PYTHONPATH=/usr/lib/python2.7/dist-packages/' >> /home/vagrant/.bashrc
