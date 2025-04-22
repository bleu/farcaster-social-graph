import { Duration, Stack, StackProps } from "aws-cdk-lib";
import * as sns from "aws-cdk-lib/aws-sns";
import * as subs from "aws-cdk-lib/aws-sns-subscriptions";
import * as sqs from "aws-cdk-lib/aws-sqs";
import { Construct } from "constructs";
import * as cdk from "aws-cdk-lib";
import { NetworkingResources } from "./networking";
import { StorageResources } from "./storage";
import { IAMResources } from "./iam";
import { ComputeResources } from "./compute";
import { SchedulerResources } from "./scheduler";

export class FarcasterInfraCdkStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // Get account and region
    const region = cdk.Stack.of(this).region;
    const account = cdk.Stack.of(this).account;

    // Initialize all resource constructs
    const networking = new NetworkingResources(this, "Networking");
    const storage = new StorageResources(this, "Storage");
    const iam = new IAMResources(this, "IAM", storage);
    const compute = new ComputeResources(
      this,
      "Compute",
      networking,
      iam,
      storage,
      region,
      account
    );
    const scheduler = new SchedulerResources(this, "Scheduler", compute);

    // Outputs
    new cdk.CfnOutput(this, "InstanceId", {
      value: compute.instance.instanceId,
      description: "EC2 Instance ID",
    });

    new cdk.CfnOutput(this, "OpFarcasterJobBucketName", {
      value: storage.jobBucket.bucketName,
      description: "S3 Bucket for job storage",
    });

    new cdk.CfnOutput(this, "EcrRepositoryUri", {
      value: storage.ecrRepository.repositoryUri,
      description: "ECR Repository URI",
    });

    new cdk.CfnOutput(this, "PushCommandExample", {
      value: `docker tag farcaster-job:latest ${storage.ecrRepository.repositoryUri}:latest && docker push ${storage.ecrRepository.repositoryUri}:latest`,
      description: "Example command to push Docker image to ECR",
    });
  }
}
