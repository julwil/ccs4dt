# Prerequisites for Windows

## Docker 
1. Install Windows subsystem for linux (WSL2 at time of writing) via Powershell
```powershell
wsl --install
```
Only works on Windows 10 versions 2004 and higher

2. Install Docker Desktop (https://www.docker.com/products/docker-desktop)

3. Ensure that you are using Linux containers, the docker menu in your taskbar should look like this ![Docker menu](scripts\docs\assets\20211030_DockerSetupWindows_Documentation.png)

4. Run commands to ensure docker is setup correctly:

- To check if docker is installed correctly and its version(at time of writing)
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