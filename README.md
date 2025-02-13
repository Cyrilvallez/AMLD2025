# AMLD 2025

## ssh key setup

Copy the key anywhere om your laptop, but usually to `~/.ssh/amld_key`.
Update the permissions on the key accordingly:

```sh
chmod 600 <path/to/the/key>
```

Add it to your ssh config:

```sh
mkdir -p ~/.ssh
vim ~/.ssh/config
```

And add the following to the file:

```sh
Host amld
  HostName <IP adress given to you>
  User ubuntu
  IdentityFile <path/to/the/key>
```

Note: to save and exit in `vim`, press `esc`, then `:wq`.
