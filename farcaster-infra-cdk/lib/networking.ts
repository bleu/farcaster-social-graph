import * as cdk from "aws-cdk-lib";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import { Construct } from "constructs";

export class NetworkingResources extends Construct {
  public readonly vpc: ec2.Vpc;
  public readonly securityGroup: ec2.SecurityGroup;

  constructor(scope: Construct, id: string) {
    super(scope, id);

    // Create a VPC
    this.vpc = new ec2.Vpc(this, "OpFarcasterJobVpc", {
      maxAzs: 2,
      natGateways: 0, // Save costs by not using NAT gateways
      subnetConfiguration: [
        {
          cidrMask: 24,
          name: "public",
          subnetType: ec2.SubnetType.PUBLIC,
        },
      ],
    });

    // Create a security group for the EC2 instance
    this.securityGroup = new ec2.SecurityGroup(
      this,
      "OpFarcasterJobSecurityGroup",
      {
        vpc: this.vpc,
        description: "Security group for scheduled job EC2 instance",
        allowAllOutbound: true, // Allow outbound traffic to download Docker images
      }
    );

    // Optional: Allow SSH access if needed for debugging
    this.securityGroup.addIngressRule(
      ec2.Peer.anyIpv4(),
      ec2.Port.tcp(22),
      "Allow SSH access from anywhere"
    );
  }
}
