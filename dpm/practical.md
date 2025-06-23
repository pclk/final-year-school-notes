1. Open cassandra.yaml in vim.

```sh
sudo vim /etc/dse/cassandra/cassandra.yaml 
```
Prevent machines from one logical cluster from joining another.
```yaml
cluster_name: "KillrVideoCluster"
```

IP address other nodes in the cluster use to access this node.
```yaml
listen_address: <NODE_IP_ADDRESS>
```

IP address that clients such as cqlsh will use to access this node.
```yaml
native_transport_address: <NODE_IP_ADDRESS>
```
don't forget the double quotes.
```yaml
- seeds: "<NODE_IP_ADDRESS>"
```

Number of VNodes this physical nodes will control.
```yaml
num_tokens: 8
```


need to learn dstat


cassandra-stress user no-warmup profile=/home/ubuntu/labwork/TestProfile.yaml ops\(insert=1,user_by_email=10\) -node ds210-node1 -rate threads=32

redo practical, and for each output, provide an explanation

if nodetool status fails, redo configuration files nad run sudo rm -rf /var/lib/cassandra/*

study level compaction
