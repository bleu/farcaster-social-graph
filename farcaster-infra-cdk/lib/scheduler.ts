import * as cdk from "aws-cdk-lib";
import * as iam from "aws-cdk-lib/aws-iam";
import * as events from "aws-cdk-lib/aws-events";
import * as targets from "aws-cdk-lib/aws-events-targets";
import * as lambda from "aws-cdk-lib/aws-lambda";
import { Construct } from "constructs";
import { ComputeResources } from "./compute";

export class SchedulerResources extends Construct {
  constructor(
    scope: Construct,
    id: string,
    computeResources: ComputeResources
  ) {
    super(scope, id);

    // Create Lambda function to start the EC2 instance
    const startInstanceFn = new lambda.Function(this, "StartInstanceFunction", {
      runtime: lambda.Runtime.NODEJS_18_X,
      handler: "index.handler",
      code: lambda.Code.fromInline(`
        const AWS = require('aws-sdk');
        const ec2 = new AWS.EC2();
        
        exports.handler = async () => {
          try {
            const params = {
              InstanceIds: ['${computeResources.instance.instanceId}'],
            };
            
            await ec2.startInstances(params).promise();
            console.log('Started instance: ${computeResources.instance.instanceId}');
            
            return {
              statusCode: 200,
              body: 'EC2 instance started successfully',
            };
          } catch (error) {
            console.error('Error starting instance:', error);
            throw error;
          }
        };
      `),
      timeout: cdk.Duration.seconds(30),
    });

    // Grant the Lambda function permissions to start EC2 instances
    startInstanceFn.addToRolePolicy(
      new iam.PolicyStatement({
        actions: ["ec2:StartInstances"],
        resources: ["*"],
      })
    );

    // Create EventBridge rule to trigger the Lambda function every week (Sunday at midnight)
    const rule = new events.Rule(this, "WeeklyJobRule", {
      schedule: events.Schedule.expression("cron(0 0 ? * SUN *)"),
      description: "Trigger weekly processing job",
    });

    // Set the Lambda function as the target of the rule
    rule.addTarget(new targets.LambdaFunction(startInstanceFn));
  }
}
