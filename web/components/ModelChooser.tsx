"use client";

import { useId, useState } from "react";
import type { ModelOption } from "@/lib/config/models";

/**
 * Picking an AI model, for a student who may recognise none of the names.
 *
 * The previous control was a flat select of 18 unfamiliar names, several
 * reading "Gemma 4 31B (open weight, free) (open weight, free tier)". Two
 * problems, and the cosmetic one was the lesser: with nothing but names, a
 * student who does not follow AI releases has no basis to choose at all.
 *
 * So the default is stated plainly with its description, and choosing is
 * opt-in. Opening the chooser groups models the way the catalog already groups
 * them and shows the one-line description we have always stored for every model
 * and never displayed.
 *
 * Native radios inside a real fieldset, and a plain button toggling a region:
 * keyboard behaviour, grouping semantics and screen reader announcements are
 * the browser's rather than reimplemented.
 */
export function ModelChooser(props: {
  options: ModelOption[];
  value: string;
  onChange: (modelId: string) => void;
  disabled?: boolean;
  /** Explains what switching affects, for example that it applies to the next
   * message rather than retroactively. */
  help?: string;
}) {
  const [open, setOpen] = useState(false);
  const panelId = useId();
  const groupNameId = useId();

  const selected = props.options.find((o) => o.id === props.value);

  // Preserve catalog order, which is category order.
  const groups: { id: string; name: string; options: ModelOption[] }[] = [];
  for (const option of props.options) {
    let group = groups.find((g) => g.id === option.groupId);
    if (!group) {
      group = { id: option.groupId, name: option.groupName, options: [] };
      groups.push(group);
    }
    group.options.push(option);
  }

  return (
    <div>
      <p className="text-sm font-bold" id={`${groupNameId}-label`}>
        Model
      </p>

      <p className="mt-1">
        <strong>{selected?.name ?? "No model selected"}</strong>
        {selected?.description ? (
          <span className="block text-sm text-dark-tan">
            {selected.description}
          </span>
        ) : null}
      </p>

      <button
        type="button"
        onClick={() => setOpen((wasOpen) => !wasOpen)}
        aria-expanded={open}
        aria-controls={panelId}
        disabled={props.disabled}
        className="mt-2 rounded-card border border-medium-tan px-3 py-1.5 text-sm font-bold hover:bg-light-tan disabled:cursor-not-allowed disabled:text-medium-gray"
      >
        {/*
          The label stays the same whether open or closed, and aria-expanded
          carries the state. A control whose accessible name changes underneath
          a screen reader user while they are focused on it is disorienting, and
          the expanded state is already announced.
        */}
        Choose a different model
      </button>

      {props.help ? (
        <p className="mt-1 text-sm text-dark-tan">{props.help}</p>
      ) : null}

      {open ? (
        <div
          id={panelId}
          className="mt-3 rounded-card border border-medium-tan bg-paper p-4"
        >
          {groups.map((group) => (
            <fieldset key={group.id} className="mt-4 first:mt-0">
              <legend className="text-sm font-bold">{group.name}</legend>
              <div className="mt-2 flex flex-col gap-3">
                {group.options.map((option) => (
                  <label key={option.id} className="flex items-start gap-2">
                    <input
                      type="radio"
                      name={`model-${groupNameId}`}
                      value={option.id}
                      checked={option.id === props.value}
                      onChange={() => props.onChange(option.id)}
                      disabled={props.disabled}
                      className="mt-1.5"
                    />
                    <span>
                      <strong>{option.name}</strong>
                      {option.recommended ? (
                        <span className="ml-2 text-sm font-bold text-miami-red">
                          suggested
                        </span>
                      ) : null}
                      <span className="block text-sm">
                        {option.description}
                      </span>
                      {option.badges.length > 0 ? (
                        // Text, not coloured chips: the information must not
                        // depend on being able to distinguish colours.
                        <span className="block text-sm text-dark-tan">
                          {option.badges.join(" · ")}
                        </span>
                      ) : null}
                    </span>
                  </label>
                ))}
              </div>
            </fieldset>
          ))}
        </div>
      ) : null}
    </div>
  );
}
