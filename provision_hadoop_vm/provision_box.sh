
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
sed -i '/^<configuration>$/ s:$:\n<property>\n\t<name>mapred.job.tracker</name>\n\t<value>localhost\:54311</value>\n\t<description>The host and port that the MapReduce job tracker runs.</description>\n</property>:' $HADOOP_HOME/conf/mapred-site.xml
sed -i '/^<configuration>$/ s:$:\n<property>\n\t<name>dfs.replication</name>\n\t<value>1</value>\n\t<description>Default block replication.</description>\n</property>:' $HADOOP_HOME/conf/hdfs-site.xml

# do sqoop
wget ftp://mirrors.sonic.net/apache/sqoop/1.4.6/sqoop-1.4.6.bin__hadoop-1.0.0.tar.gz
tar -zxvf sqoop-1.4.6.bin__hadoop-1.0.0.tar.gz
#clean up
sudo rm *.tar.*
sudo chown vagrant:vagrant *
#edit .bashrc
echo 'source /etc/environment' >> /home/vagrant/.bashrc
