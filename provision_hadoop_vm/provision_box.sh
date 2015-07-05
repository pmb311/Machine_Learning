# do maven
wget ftp://mirror.reverse.net/pub/apache/maven/maven-3/3.1.1/binaries/apache-maven-3.1.1-bin.tar.gz
tar -xvf apache-maven-3.1.1-bin.tar.gz
mv apache-maven-3.1.1 maven
# create final env settings centos
sudo sh -c "cat >> /etc/environment" <<'EOF'
export HADOOP_HOME=/home/vagrant/hadoop-1.2.1
export MAVEN_HOME=/home/vagrant/maven
export JAVA_HOME=/usr/lib/jvm/java-6-openjdk/jre
PATH=$PATH:$JAVA_HOME/bin:$MAVEN_HOME/bin:$HADOOP_HOME/bin
export PATH
EOF
# set env var's
source /etc/environment
# do sqoop
wget ftp://mirrors.sonic.net/apache/sqoop/1.4.6/sqoop-1.4.6.bin__hadoop-1.0.0.tar.gz
tar -zxvf sqoop-1.4.6.bin__hadoop-1.0.0.tar.gz
#clean up
sudo rm *.tar.*
sudo chown vagrant:vagrant *
# ssh-keygen -t dsa -P '' -f /home/vagrant/.ssh/id_dsa
# cat /home/vagrant/.ssh/id_dsa.pub >> /home/vagrant/.ssh/authorized_keys
