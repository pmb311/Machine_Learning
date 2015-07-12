group { "puppet":
ensure => "present",
}
Exec { path => [ "/bin/", "/sbin/" , "/usr/bin/", "/usr/sbin/" ] }
$hadoop_home = "/home/vagrant"
exec { 'apt-get update':
command => 'apt-get update',
}
package { "openjdk-6-jdk" :
ensure => present,
require => Exec['apt-get update'],
}
exec { "download_hadoop": 
command => "wget -O /home/vagrant/hadoop­1.2.1.tar.gz http://apache.mirrors.pair.com/hadoop/common/hadoop-1.2.1/hadoop-1.2.1-bin.tar.gz", 
path => $path, 
unless => "ls /home/vagrant | grep hadoop­1.2.1"
} 
 
exec { "unpack_hadoop" : 
  command => "tar -zxf /home/vagrant/hadoop­1.2.1.tar.gz", 
  path => $path, 
  creates => "${hadoop_home}/hadoop­1.2.1", 
  require => Exec["download_hadoop"] 
}

$python_packages = [ "python-numpy", "python-scipy", "python-matplotlib", "ipython", "ipython-notebook", "python-pandas", "python-sympy", "python-nose" ]
package { $python_packages:
	ensure => installed,
	require => Exec['apt-get update'],
}

package { "emacs23":
	ensure => installed,
	require => Exec['apt-get update'],
}