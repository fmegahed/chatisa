import { ALLOWED_EMAIL_DOMAIN } from "@/lib/auth/domain";

/**
 * What the members API accepts for a teammate invite (a valid address on
 * the Miami domain, at most 200 characters), stated up front and checked in
 * the browser so the field can explain a problem before anything is sent
 * (#26). The API route stays the authority.
 */
export const INVITE_EMAIL_HINT = `Use the ${ALLOWED_EMAIL_DOMAIN} address they sign in with.`;

export function inviteEmailError(value: string): string | null {
  const email = value.trim().toLowerCase();
  if (!email) return "Enter your teammate's email address.";
  if (email.length > 200 || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
    return "Enter a valid email address.";
  }
  if (!email.endsWith(`@${ALLOWED_EMAIL_DOMAIN}`)) {
    return `Use their ${ALLOWED_EMAIL_DOMAIN} address. Other email addresses cannot be added.`;
  }
  return null;
}
