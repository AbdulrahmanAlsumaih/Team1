# Connecting to the Cloud GPU Server

This guide will walk you through the steps to connect to the Ubuntu GPU server I have provisioned.

## System Specifications
```
---------------------------------------------------------------
|  CPU     : AMD Ryzen 7 7800X3D (8-core, 4.2 GHz, Zen 4)      |
|  GPU     : NVIDIA GeForce RTX 4090, 24 GB VRAM               |
|  Memory  : 64 GB RAM                                         |
|  Storage : 2 TB SSD                                          |
|  OS      : Ubuntu 20.04.6 LTS                                |
---------------------------------------------------------------
```

## Prerequisites

- **SSH Client**: Ensure you have an SSH client installed. Most Linux and macOS systems come with SSH pre-installed. Windows users can use tools like **PuTTY** or the built-in **OpenSSH** client (available on Windows 10 and later).
- **Credentials**: Contact the owner to obtain the necessary credentials.
- **VPN Network Access**: This needs to be approved by the admin [request here](https://login.tailscale.com/uinv/i05bd69e23b391775)

## Steps to Connect

### 1. Connect to Tailscale VPN

Make sure you are connected to Tailscale VPN.

### 2. Open Terminal or SSH Client

On macOS or Linux, open the Terminal application. On Windows, either open **Command Prompt**, **PowerShell**, or **PuTTY** (if installed).

### 3. Connect Using SSH

To connect using your username, run the following command:

```bash
ssh -p 2222 username@100.111.139.62
```

Replace `username` with your actual username.

### 4. Enter Password

SSH will now prompt you to enter the password. Type your password and press **Enter**.

> Note: When entering your password, you won't see any characters. This is a security feature.

### 5. Accept the Server's Fingerprint

On your first connection, SSH will ask you to confirm the server's fingerprint. Type `yes` and press **Enter** to add the server to your known hosts file.

```plaintext
Are you sure you want to continue connecting (yes/no/[fingerprint])? yes
```

### 6. Successful Connection

If everything is correct, you will now be connected to your Ubuntu server. You should see a command prompt for the server, which might look something like this:

```plaintext
Welcome to Ubuntu 20.04 LTS (GNU/Linux 5.4.0-26-generic x86_64)
ubuntu@your_server_ip:~$
```

## Additional Tips

- **Transfer Files Using SCP**: To copy files to or from the server, use `scp` (Secure Copy Protocol). Here’s an example of copying a file from your local machine to the server:

  ```bash
  scp /path/to/local/file username@your_server_ip:/path/to/remote/directory
  ```

- **Exit the SSH Session**: When you’re done, type `exit` and press **Enter** to close the SSH connection.

## Troubleshooting

- **Connection Timeout**: If you can’t connect, ensure the vpn is connected, the credentials are correct, and that your network allows SSH connections (port 2222).
- **Permission Denied**: This error typically means your username or password. Double-check the credentials.
- **Firewall Settings**: Ensure the client's firewall allows outbound SSH connections.

This guide should help you connect to my Ubuntu SSH server. If you encounter issues, verify each step and consult me for further assistance.
