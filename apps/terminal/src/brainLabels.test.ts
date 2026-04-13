import { describe, expect, it } from "vitest";

import { translateBrainIdentifier } from "./brainLabels.js";
import { translateActionName } from "./displayLabels.js";

describe("TLH terminology alignment", () => {
  it("translates TLH runtime contributors into brain-science Chinese labels", () => {
    expect(translateBrainIdentifier("InstinctField")).toBe("本能场");
    expect(translateBrainIdentifier("BodyStateAgent")).toBe("躯体内感监测");
  });

  it("translates TLH actions into aligned Chinese labels", () => {
    expect(translateActionName("absorb")).toBe("吸收沉积");
    expect(translateActionName("monologue")).toBe("内心独白");
    expect(translateActionName("nothing")).toBe("保持静默");
    expect(translateActionName("die")).toBe("自主结束生命");
  });
});
