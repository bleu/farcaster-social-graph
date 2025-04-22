export const ubuntuArmAmi = {
  "us-east-2": "ami-0560690593473ded1",
} as const;

// Return correct AMI for a region
export function getUbuntuArmAmi(region: string): string {
  const ami = (ubuntuArmAmi as Record<string, string>)[region];
  if (!ami) {
    throw new Error(`No Ubuntu ARM AMI defined for region ${region}`);
  }
  return ami;
}
