#!/usr/bin/env node
import * as cdk from "aws-cdk-lib";
import { FarcasterInfraCdkStack } from "../lib/farcaster-infra-cdk-stack";

const app = new cdk.App();
new FarcasterInfraCdkStack(app, "FarcasterInfraCdkStack", {
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT || "123456789012",
    region: process.env.CDK_DEFAULT_REGION || "us-east-1",
  },
});

cdk.Tags.of(app).add("Project", "OpFarcasterSocialGraph");
cdk.Tags.of(app).add("Environment", "Production");
