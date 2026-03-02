type Props = {
  selected: number;
  setSelected: React.Dispatch<React.SetStateAction<number>>;
}
const RangeSelector = ({selected, setSelected} : Props) => {
  return (
     <div className="h-5 p-0.5 border border-[#282735] rounded-sm flex gap-0.5 items-center justify-center">
          <button className={"min-w-4 h-4 px-0.5 cursor-pointer text-[8px] text-center font-semibold rounded-xs text-[#11101A]" + (selected === 0 ? " purple-gradient" : " text-[#282735]")} onClick={() => setSelected(0)}>
            1D
          </button>
          <button className={"min-w-4 h-4 px-0.5 cursor-pointer text-[8px] text-center font-semibold rounded-xs text-[#11101A]" + (selected === 1 ? " purple-gradient" : " text-[#282735]")} onClick={() => setSelected(1)}>
            7D
          </button>
          <button className={"min-w-4 h-4 px-0.5 cursor-pointer text-[8px] text-center font-semibold rounded-xs text-[#11101A]" + (selected === 2 ? " purple-gradient" : " text-[#282735]")} onClick={() => setSelected(2)}>
            1M
          </button>
          <button className={"min-w-4 h-4 px-0.5 cursor-pointer text-[8px] text-center font-semibold rounded-xs text-[#11101A]" + (selected === 3 ? " purple-gradient" : " text-[#282735]")} onClick={() => setSelected(3)}>
            3M
          </button>
          <button className={"min-w-4 h-4 px-0.5 cursor-pointer text-[8px] text-center font-semibold rounded-xs text-[#11101A]" + (selected === 4 ? " purple-gradient" : " text-[#282735]")} onClick={() => setSelected(4)}>
            1Y
          </button>
          <button className={"min-w-4 h-4 px-0.5 cursor-pointer text-[8px] text-center font-semibold rounded-xs text-[#11101A]" + (selected === 5 ? " purple-gradient" : " text-[#282735]")} onClick={() => setSelected(5)}>
            ALL
          </button>
    </div>
  )
}

export default RangeSelector