import { useState } from "react";
import RangeSelector from "./RangeSelector";
import Graph from "./Graph";

const EquityCurve = () => {
  const [selected, setSelected] = useState(4);
  return (
    <div className="w-120 h-75 rounded-lg purple-shadow p-5">
      <div className="flex justify-between items-center mb-8 w-full">
        <div>
          <div className="text-white font-semibold text-md">Buy & Hold Curve</div>
          <div className="text-[#BCBCBC] text-[12px]">SPY</div>
        </div>
        <RangeSelector selected={selected} setSelected={setSelected}></RangeSelector>
      </div>
      <div className="w-108 h-49">
        <Graph mode={selected}></Graph>
      </div>      
    </div>
  )
}

export default EquityCurve