# Parameter Golf on AWS: p5.48xlarge Spot Instance Setup

## Pre-requisites

- AWS account with P instance quota in eu-north-1 (192 vCPUs for p5.48xlarge)
- `aws` CLI installed and configured (`aws configure`)
- SSH key pair in eu-north-1

## Instance Specs

| | |
|---|---|
| **Instance** | p5.48xlarge |
| **GPUs** | 8x NVIDIA H100 SXM (80GB each, 640GB total) |
| **vCPUs** | 192 |
| **RAM** | 2,048 GiB |
| **Storage** | 3,800 GiB NVMe SSD |
| **Spot price** | ~$12.96/hr (eu-north-1) |
| **On-demand** | ~$56.88/hr |

## One-Time Setup (run locally)

### 1. Create a Key Pair

Skip if you already have one in eu-north-1.

```bash
aws ec2 create-key-pair \
  --key-name parameter-golf \
  --region eu-north-1 \
  --query 'KeyMaterial' \
  --output text > ~/.ssh/parameter-golf.pem

chmod 600 ~/.ssh/parameter-golf.pem
```

### 2. Create a Security Group

```bash
SG_ID=$(aws ec2 create-security-group \
  --group-name parameter-golf-sg \
  --description "Parameter Golf - SSH access" \
  --region eu-north-1 \
  --query 'GroupId' --output text)

aws ec2 authorize-security-group-ingress \
  --group-id "$SG_ID" \
  --protocol tcp --port 22 --cidr 0.0.0.0/0 \
  --region eu-north-1

echo "Security Group: $SG_ID"
```

Save the `$SG_ID` — you'll use it to launch instances.

## Launching a Spot Instance

### 1. Find the Latest Deep Learning AMI

```bash
AMI_ID=$(aws ec2 describe-images \
  --region eu-north-1 \
  --owners amazon \
  --filters \
    "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2*Ubuntu 22.04*" \
    "Name=state,Values=available" \
  --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
  --output text)

echo "AMI: $AMI_ID"
```

### 2. Launch the Instance

```bash
INSTANCE_ID=$(aws ec2 run-instances \
  --region eu-north-1 \
  --instance-type p5.48xlarge \
  --image-id "$AMI_ID" \
  --key-name parameter-golf \
  --security-group-ids "$SG_ID" \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":200,"VolumeType":"gp3","Iops":6000,"Throughput":400}}]' \
  --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate"}}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=parameter-golf}]' \
  --query 'Instances[0].InstanceId' \
  --output text)

echo "Instance: $INSTANCE_ID"
```

### 3. Wait for it and Get the IP

```bash
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region eu-north-1

PUBLIC_IP=$(aws ec2 describe-instances \
  --instance-ids "$INSTANCE_ID" \
  --region eu-north-1 \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text)

echo "SSH: ssh -i ~/.ssh/parameter-golf.pem ubuntu@$PUBLIC_IP"
```

### 4. SSH In

```bash
ssh -i ~/.ssh/parameter-golf.pem ubuntu@$PUBLIC_IP
```

## After SSH-ing In

Run the setup script (does everything):

```bash
curl -sL https://raw.githubusercontent.com/openai/parameter-golf/main/setup_aws.sh | bash
```

Or see `setup_aws.sh` for what it does step by step.

## Running Autoresearch

Start a tmux session (survives SSH disconnects):

```bash
tmux new -s autoresearch
cd ~/parameter-golf
```

Launch Claude Code:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
claude
```

Tell it to read `program.md` and start the experiment loop. The agent will iterate autonomously: modify `train_gpt.py`, train for ~10 min on 8xH100, check val_bpb, keep or discard, repeat.

Detach from tmux: `Ctrl+B` then `D`. Reattach: `tmux attach -t autoresearch`.

## Instance Management

**Stop (preserves EBS, stops billing for compute but not storage):**

```bash
# Can't stop spot instances — they can only be terminated.
# Use on-demand if you need stop/start behavior.
```

**Terminate (done for the day):**

```bash
aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region eu-north-1
```

**Check running instances:**

```bash
aws ec2 describe-instances \
  --region eu-north-1 \
  --filters "Name=tag:Name,Values=parameter-golf" "Name=instance-state-name,Values=running" \
  --query 'Reservations[].Instances[].[InstanceId,PublicIpAddress,LaunchTime]' \
  --output table
```

## Cost Tracking

| Item | Cost |
|------|------|
| Spot price | ~$12.96/hr |
| Per experiment (~15 min) | ~$3.25/run |
| Overnight (8 hrs, ~32 experiments) | ~$104 |
| Full day (24 hrs, ~96 experiments) | ~$311 |

## Spot Instance Notes

- AWS can reclaim spot instances with **2 minutes warning**
- For 10-min training runs, preemption risk per run is low
- If preempted mid-run, you lose that run only — prior git commits are safe
- Autoresearch handles crashes gracefully (logs crash, reverts, moves on)
- Always use `tmux` so SSH disconnects don't kill the agent
- **Terminate the instance when you're done** — spot instances can't be stopped

## Troubleshooting

**InsufficientInstanceCapacity:** Spot capacity unavailable. Try again later or try a different AZ:

```bash
# Try specific AZ
--placement '{"AvailabilityZone":"eu-north-1b"}'
```

**SpotMaxPriceTooLow:** Spot price spiked above on-demand. Wait or switch region.

**Fallback regions** (if eu-north-1 has no capacity):
- us-east-1 (Virginia) — largest region, best availability
- us-west-2 (Oregon) — good H100 availability
- eu-west-1 (Ireland) — EU alternative
