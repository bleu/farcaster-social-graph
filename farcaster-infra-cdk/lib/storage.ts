import * as cdk from "aws-cdk-lib";
import * as s3 from "aws-cdk-lib/aws-s3";
import * as ecr from "aws-cdk-lib/aws-ecr";
import { Construct } from "constructs";

export class StorageResources extends Construct {
  public readonly jobBucket: s3.Bucket;
  public readonly ecrRepository: ecr.Repository;

  constructor(scope: Construct, id: string) {
    super(scope, id);

    // Create S3 bucket to store job artifacts and results
    this.jobBucket = new s3.Bucket(this, "OpFarcasterJobBucket", {
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: false,
      versioned: true,
    });

    // Create ECR repository for Docker images
    this.ecrRepository = new ecr.Repository(this, "OpFarcasterJobRepository", {
      repositoryName: "farcaster-job-repository",
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      lifecycleRules: [
        {
          maxImageCount: 5,
          description: "Only keep the 5 most recent images",
        },
      ],
    });
  }
}
