# Welcome to your CDK TypeScript project

You should explore the contents of this project. It demonstrates a CDK app with an instance of a stack (`FarcasterInfraCdkStack`)
which contains an Amazon SQS queue that is subscribed to an Amazon SNS topic.

The `cdk.json` file tells the CDK Toolkit how to execute your app.

## Useful commands

- `npm run build` compile typescript to js
- `npm run watch` watch for changes and compile
- `npm run test` perform the jest unit tests
- `cdk deploy` deploy this stack to your default AWS account/region
- `cdk diff` compare deployed stack with current state
- `cdk synth` emits the synthesized CloudFormation template

# Farcaster Social Graph Scheduled Job

This CDK project sets up a scheduled EC2 job that runs the Farcaster social graph processing container once per week. The infrastructure:

1. Uses EventBridge to trigger a Lambda function every Sunday at midnight
2. The Lambda starts a pre-configured EC2 instance
3. The EC2 instance pulls the Docker image from ECR and runs it
4. After completion, the instance automatically shuts down to save costs

## Architecture

The infrastructure is organized into modular constructs:

- **NetworkingResources**: VPC and security groups
- **StorageResources**: S3 bucket and ECR repository
- **IAMResources**: IAM roles and permissions
- **ComputeResources**: EC2 instance configuration
- **SchedulerResources**: Lambda and EventBridge scheduling

### Components:

- **Amazon ECR**: Stores the Docker image for the job
- **Amazon EC2**: r6i.8xlarge instance with high memory for data processing
- **Amazon S3**: Stores job artifacts and results
- **AWS Lambda**: Triggered by EventBridge to start the EC2 instance
- **Amazon EventBridge**: Schedules the weekly job execution
- **AWS IAM**: Manages permissions for all components

## Prerequisites

- Node.js 18.x or later
- AWS CLI configured with appropriate credentials
- AWS CDK installed globally (`npm install -g aws-cdk`)
- Docker installed locally (for building and pushing the image)

## Setup

1. Install dependencies:

```bash
npm install
```

2. Customize the configuration:

- Update the ECR repository name in `StorageResources` if needed
- Adjust the EC2 instance type in `ComputeResources` as necessary
- Set your SSH key pair name for EC2 access

3. Bootstrap your AWS environment (if you haven't already):

```bash
cdk bootstrap
```

4. Deploy the stack:

```bash
cdk deploy
```

5. Build and push your Docker image to ECR:

```bash
# Login to ECR
aws ecr get-login-password --region YOUR_REGION | docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.YOUR_REGION.amazonaws.com

# Build the Docker image
docker build -t farcaster-job .

# Tag and push to ECR (the exact command will be output after deploying the stack)
docker tag farcaster-job:latest YOUR_ECR_REPOSITORY_URI:latest
docker push YOUR_ECR_REPOSITORY_URI:latest
```

## Customization Options

- **Instance Type**: Change the EC2 instance type in `ComputeResources` based on your workload
- **Schedule**: Modify the cron expression in `SchedulerResources` to change the schedule
- **Storage**: Adjust the EBS volume size and IOPS as needed
- **Security**: Review and modify the security group rules in `NetworkingResources`
- **Safety Timeout**: Adjust the 4-hour safety timeout rule if your job needs more time

## Monitoring

- Check CloudWatch Logs for Lambda and EC2 execution logs
- Instance termination is logged to the S3 bucket
- The EC2 instance tags show the current job status

## Cost Optimization

This solution is designed to be cost-effective:

- The EC2 instance runs only when needed (likely 2-3 hours per week)
- All resources except S3 bucket and ECR repository will be deleted when stack is removed
- Instance automatically terminates after job completion
- Safety mechanism ensures instance is stopped even if job fails

## Cleanup

To remove all resources except the S3 bucket and ECR repository:

```bash
cdk destroy
```

## Troubleshooting

- If the job fails, SSH into the instance before it terminates by setting longer timeout
- Check the CloudWatch logs for the EC2 instance
- Verify Docker container logs with `docker logs`
- Check the ECR repository to ensure your image is properly pushed
