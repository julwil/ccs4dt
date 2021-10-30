# Prerequisites for Windows

## Docker and WSL2
1. Install Windows subsystem for linux (WSL2 at time of writing) via Powershell
```powershell
wsl --install
```
Only works on Windows 10 versions 2004 and higher

2. Install Docker Desktop (https://www.docker.com/products/docker-desktop)

3. Ensure that you are using Linux containers, the docker menu in your taskbar should look like this ![Docker menu](scripts\docs\assets\20211030_DockerSetupWindows_Documentation.png)

4. Run commands to ensure docker is setup correctly:

- To check if docker is installed correctly and its version (at time of writing)
```console
docker --version
Docker version 20.10.8, build 3967b7d
```
- To check if docker-compose is installed correctly and its version (at time of writing)
```console
docker-compose --version
Docker Compose version v2.0.0
```
- To check if docker desktop is running currently on your windows machine 
```console
wsl -l -v 
  NAME                   STATE           VERSION
  docker-desktop-data    Running         2      
  docker-desktop         Running         2    
```

# Executing commands inside the docker container 

## Starting up the docker container
To start up the docker container run the following command inside the ccs4dt folder:

```console
..\ccs4dt> docker-compose up
```

You should get output that indicates that all containers are setup and running 

```console
[+] Running 3/3
 - Network ccs4dt_ccs4dt     Started  
 - Container api     Started  
 - Container influxdb     Started  
 ```   

## Accessing commands inside the container

To switch your terminal to inside the container execute the following command (while the docker container)

```console
docker exec -it api /bin/bash
```

## Running tests

To execute the full testsuite run the following command **inside the docker container**

```console
pytest
```

To run only selective tests run the following command **based on the module for which you would like to execute the tests**
```console
pytest [path-to-module-tests]
```
e.g. synthetic_data_generation:
```console
pytest pytest scripts/synthetic_data_generation/tests
```
