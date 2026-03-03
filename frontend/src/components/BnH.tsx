import { useEffect, useState } from "react";
import Graph from "./Graph";
import RangeSelector from "./RangeSelector";
import dummyData from "./dummyEquity.json";
import { fetchBnHData } from "../lib/fetchData";

const BnH = () => {
  const [selected, setSelected] = useState(4);
  const [data, setData] = useState<{date: string, equity: number}[]>([]);

  useEffect(() => { 
    const fetchData = async () => {
      try {
        const rawData = await fetchBnHData();
        console.log('first', rawData);
        setData(rawData);
        if (!rawData || rawData.length === 0) {
          setData(dummyData);
        }
        console.log('rawData', rawData)

      } catch (error) {
        console.error("Error fetching equity data:", error);
      }
    }
    fetchData();

  }, [])

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
        <Graph mode={selected} data={data}></Graph>
      </div>      
    </div>
  )
}

export default BnH