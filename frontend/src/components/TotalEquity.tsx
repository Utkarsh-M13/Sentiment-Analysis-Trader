import { useEffect, useState } from "react";
import { get_equity_now_and_reference } from "../lib/fetchData";

const InnerDiv = ({ thisWeekChange } : { thisWeekChange: number }) => {
    const v = thisWeekChange;
    if (v === 0) {
      return <span className="text-gray-500 font-semibold text-sm">0.00%</span>;
    } else if (v > 0) {
      return (
        <span className="text-green-500 font-semibold text-sm">
          +{v.toFixed(2)}%
        </span>
      );
    } else {
      return (
        <span className="text-red-500 text-sm">
          {v.toFixed(2)}%
        </span>
      );
    }
  }

const TotalEquity = () => {
  const dummyEquity = 0;
  function formatNumber(num: number): string {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  }
  const [equity, setEquity] = useState(dummyEquity);
  const [thisWeekChange, setThisWeekChange] = useState(0);

  useEffect(() => {
    // Simulate fetching equity data with a delay
    const fetchEquity = async () => {
      try {
        // Replace this with your actual API call to fetch equity data
        const raw = await get_equity_now_and_reference();
        console.log('raw equity data', raw);
        const fetchedEquity = raw ? parseFloat(raw.current_equity.toFixed(4)) : dummyEquity;
        const thisWeekChange = raw ? parseFloat(((raw.current_equity - raw.reference_equity) / raw.reference_equity * 100).toFixed(4)) : 0;
        setThisWeekChange(thisWeekChange);
        setEquity(fetchedEquity);
      } catch (error) {
        console.error("Error fetching equity data:", error);
      }
    };

    fetchEquity();
  }, []);

  

  return (
    <div className="w-40 h-30 rounded-lg purple-shadow text-white p-4 text-center">
      <div className="font-semibold text-[16px]">Total Equity</div>
      <div className="font-semibold text-2xl mt-2">${formatNumber(equity)}</div>
      <div className="text-xs mt-2 text-[#9F9F9F]">
        <InnerDiv thisWeekChange={thisWeekChange} ></InnerDiv>
      </div>
    </div>
  )
}

export default TotalEquity