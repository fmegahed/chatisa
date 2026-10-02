/**
 * Visible required-field marker (WCAG 3.3.2). The asterisk is hidden from
 * screen readers because the input itself carries `required`, which they
 * announce; pair every marked form with <RequiredNote /> so sighted users
 * know what the asterisk means.
 */
export function RequiredMark() {
  return (
    <span aria-hidden="true" className="ml-0.5 font-bold text-miami-red">
      *
    </span>
  );
}

export function RequiredNote({ className = "" }: { className?: string }) {
  return (
    <p className={`text-sm text-dark-tan ${className}`}>
      Fields marked <span className="font-bold text-miami-red">*</span> are
      required.
    </p>
  );
}
