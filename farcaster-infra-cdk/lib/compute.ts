import * as cdk from "aws-cdk-lib";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import { Construct } from "constructs";
import { NetworkingResources } from "./networking";
import { IAMResources } from "./iam";
import { StorageResources } from "./storage";
import { getUbuntuArmAmi } from "./ami-mappings";

export class ComputeResources extends Construct {
  public readonly instance: ec2.Instance;

  constructor(
    scope: Construct,
    id: string,
    networking: NetworkingResources,
    iamResources: IAMResources,
    storage: StorageResources,
    region: string,
    account: string
  ) {
    super(scope, id);

    // Create user data to run on EC2 instance startup
    const userData = ec2.UserData.forLinux();

    // Get the account specific ECR URI
    const ecrRepositoryUri = storage.ecrRepository.repositoryUri;

    // Add Ubuntu-specific commands (replacing yum with apt)
    userData.addCommands(
      // Update package list and install dependencies
      "apt-get update -y",
      "apt-get install -y apt-transport-https ca-certificates curl software-properties-common",
      "curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -",
      'add-apt-repository "deb [arch=arm64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"',
      "apt-get update -y",
      "apt-get install -y docker-ce docker-ce-cli containerd.io awscli",
      "systemctl start docker",
      "systemctl enable docker",

      // Create directories for data
      "mkdir -p /home/ubuntu/data/raw /home/ubuntu/data/interim /home/ubuntu/data/checkpoints /home/ubuntu/data/models",
      "chmod -R 777 /home/ubuntu/data",
      "chown -R ubuntu:ubuntu /home/ubuntu/data",

      // Login to ECR
      `aws ecr get-login-password --region ${region} | docker login --username AWS --password-stdin ${account}.dkr.ecr.${region}.amazonaws.com`,

      // Pull and run the Docker image
      `docker pull ${ecrRepositoryUri}:latest`,
      `docker run --rm -v /home/ubuntu/data:/app/data ${ecrRepositoryUri}:latest`,

      // Log completion
      'echo "OpFarcasterJob completed at $(date)" | tee /home/ubuntu/job-complete.log',

      // Shutdown instance after completion
      "shutdown -h now"
    );

    // Get AMI for the current region
    const amiId = "ami-0560690593473ded1"; // getUbuntuArmAmi(region);

    this.instance = new ec2.Instance(this, "OpFarcasterJobInstance", {
      vpc: networking.vpc,
      // Choose a Graviton (ARM) instance for better price-performance
      instanceType: ec2.InstanceType.of(
        ec2.InstanceClass.R6G, // ARM-based Graviton processor
        ec2.InstanceSize.XLARGE8
      ),
      // Use custom Ubuntu ARM AMI
      machineImage: ec2.MachineImage.genericLinux({
        [region]: amiId,
      }),
      securityGroup: networking.securityGroup,
      role: iamResources.ec2Role,
      userData,
      blockDevices: [
        {
          deviceName: "/dev/sda1", // Ubuntu uses /dev/sda1 instead of /dev/xvda
          volume: ec2.BlockDeviceVolume.ebs(200, {
            // Adjust volume size as needed
            volumeType: ec2.EbsDeviceVolumeType.GP3,
            iops: 6000, // Higher IOPS for better performance
            throughput: 250,
          }),
        },
      ],
    });

    // Add Name tag to instance
    cdk.Tags.of(this.instance).add("Name", "OpFarcasterJobInstance");

    // Stop instance if running for more than 4 hours (safety mechanism)
    // First, create the rule without the target
    const shutdownRule = new cdk.CfnResource(this, "InstanceShutdownRule", {
      type: "AWS::Events::Rule",
      properties: {
        Description:
          "Rule to shut down instance after 4 hours as a safety measure",
        State: "ENABLED",
        ScheduleExpression: "rate(4 hours)",
      },
    });

    // Then, create the target as a separate resource that depends on both the rule and the instance
    const ruleTarget = new cdk.CfnResource(this, "ShutdownRuleTarget", {
      type: "AWS::Events::Target",
      properties: {
        Rule: shutdownRule.ref,
        Arn: `arn:aws:ssm:${region}:${account}:automation-definition/AWS-StopEC2Instance`,
        Id: "StopInstanceTarget",
        Input: `{"InstanceId":["${this.instance.instanceId}"]}`,
        RoleArn: iamResources.eventsRole.roleArn,
      },
    });

    // Add explicit dependency
    ruleTarget.node.addDependency(this.instance);
  }
}
