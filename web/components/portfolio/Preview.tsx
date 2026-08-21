"use client";

/** The rendered site, shown exactly as it will be published. The empty
 * sandbox attribute keeps the preview from running scripts or reaching the
 * parent page. */
export function Preview(props: { html: string }) {
  return (
    <iframe
      sandbox=""
      srcDoc={props.html}
      title="Site preview"
      className="h-[36rem] w-full rounded-card border border-medium-tan bg-white"
    />
  );
}
