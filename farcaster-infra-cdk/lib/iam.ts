import * as cdk from "aws-cdk-lib";
import * as iam from "aws-cdk-lib/aws-iam";

import { Construct } from "constructs";
import { StorageResources } from "./storage";

export class IAMResources extends Construct {
  public readonly ec2Role: iam.Role;
  public readonly eventsRole: iam.Role;

  constructor(scope: Construct, id: string, storage: StorageResources) {
    super(scope, id);

    // Create IAM role for the EC2 instance
    this.ec2Role = new iam.Role(this, "OpFarcasterJobEc2Role", {
      assumedBy: new iam.ServicePrincipal("ec2.amazonaws.com"),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName(
          "AmazonSSMManagedInstanceCore"
        ),
      ],
    });

    // Grant S3 bucket access to EC2 role
    storage.jobBucket.grantReadWrite(this.ec2Role);

    // Grant ECR access to EC2 role
    storage.ecrRepository.grantPull(this.ec2Role);

    // Create EventBridge role for stopping instances
    this.eventsRole = new iam.Role(this, "EventRole", {
      assumedBy: new iam.ServicePrincipal("events.amazonaws.com"),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName("AmazonSSMFullAccess"),
      ],
    });
  }
}
